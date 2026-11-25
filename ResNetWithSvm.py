from pathlib import Path
import os
import argparse
from collections import Counter
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from ResNet import Block, ResNet

# Settings
SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
CHECKPOINT = Path(SCRIPT_PATH) / "model34_20261120_113410_4"   # the trained ResNet to start from
EPOCHS = 10
BATCH_SIZE = 32
RESNET_LR = 0.001            # small nudges so the trained ResNet isn't wrecked
SVM_LR = 0.01                # bigger nudges for the new SVM layer
REGULARIZATION = 0.001       # L2 penalty on the SVM weights
RESNET_WEIGHT_DECAY = 1e-4   # L2 penalty on the ResNet weights, stops it inflating the features
data_dir = os.path.join(SCRIPT_PATH, "spectrograms")


parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--train_top_20", action="store_true")
args = parser.parse_args()

# Check that the checkpoint matches the chosen mode
checkpoint_weights = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
checkpoint_classes = checkpoint_weights["fc.weight"].shape[0]
top_20_mode = args.train_top_20

if top_20_mode and checkpoint_classes != 20:
    raise SystemExit(
        f"{CHECKPOINT.name} was trained on {checkpoint_classes} classes, not the top 20.\n"
        f"Run without --train_top_20, or set CHECKPOINT to a top-20 checkpoint."
    )
if not top_20_mode and checkpoint_classes == 20:
    raise SystemExit(
        f"{CHECKPOINT.name} was trained on the top 20 classes.\n"
        f"Run with --train_top_20, or set CHECKPOINT to an all-species checkpoint."
    )

dataset = datasets.ImageFolder(data_dir, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))

class_sizes = Counter(dataset.targets)

if top_20_mode:
    usable_class_ids = sorted(class_id for class_id, count in class_sizes.most_common(20))
else:
    usable_class_ids = sorted(
        class_id for class_id, count in class_sizes.items() if count > 10
    )

old_to_new = {old_id: new_id for new_id, old_id in enumerate(usable_class_ids)}
class_names = [dataset.classes[old_id] for old_id in usable_class_ids]
num_classes = len(usable_class_ids)

usable_indices = [
    index for index, target in enumerate(dataset.targets) if target in old_to_new
]

mapped_labels = [old_to_new[dataset.targets[index]] for index in usable_indices]


class RemappedSubset(Dataset):
    def __init__(self, image_dataset, indices, label_map):
        self.image_dataset = image_dataset
        self.indices = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_index = self.indices[position]
        image, old_label = self.image_dataset[original_index]
        return image, self.label_map[old_label]


train_indexes, temp_indexes, train_labels, temp_labels = train_test_split(
    usable_indices,
    mapped_labels,
    train_size=0.7,
    random_state=42,
    stratify=mapped_labels,
)

test_indexes, validate_indexes = train_test_split(
    temp_indexes,
    train_size=0.67,
    random_state=42,
    stratify=temp_labels,
)

train_data = RemappedSubset(dataset, train_indexes, old_to_new)
test_data = RemappedSubset(dataset, test_indexes, old_to_new)
validate_data = RemappedSubset(dataset, validate_indexes, old_to_new)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)        # new order every epoch
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
validate_loader = DataLoader(validate_data, batch_size=BATCH_SIZE, shuffle=False)


class ResNetWithSVM(nn.Module):
    """ResNet makes the 512 features, the SVM layer on top classifies them."""
    def __init__(self, resnet, num_classes):
        super().__init__()
        self.resnet = resnet
        self.resnet.fc = nn.Identity()            # remove the ResNet's own classifier
        self.svm = nn.Linear(512, num_classes)    # the SVM: one scorecard per species

    def forward(self, images):
        features = self.resnet.forward_features(images)  # [batch, 512]
        return self.svm(features)                        # [batch, num_classes]


hinge_loss = nn.MultiMarginLoss(p=1, margin=1.0)  # correct species must win by >= 1


def svm_loss(model, scores, labels):
    """Hinge loss + weight penalty, the same as train_svm in SVM.py"""
    weight_penalty = model.svm.weight.square().sum()
    return hinge_loss(scores, labels) + REGULARIZATION * weight_penalty


def evaluate(model, loader, device):
    """Returns average hinge loss, all predictions and all true labels."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            total_loss += hinge_loss(scores, labels).item()
            all_preds.append(scores.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
    return total_loss / len(loader), torch.cat(all_preds), torch.cat(all_labels)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    print(f"Checkpoint was trained on {checkpoint_classes} classes "
          f"-> {'top 20' if top_20_mode else 'all species'} mode, SVM classifies {num_classes} species")
    print(f"Train: {len(train_data)}  Test: {len(test_data)}  Validate: {len(validate_data)}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    SAVE_PATH = Path(SCRIPT_PATH) / f"resnet_svm_{'top20' if top_20_mode else 'all'}_{timestamp}.pt"

    # 1. Build the ResNet with the checkpoint's class count so the weights fit, then load them
    resnet = ResNet(Block, [3, 4, 6, 3], 3, checkpoint_classes)
    resnet.load_state_dict(checkpoint_weights)

    # 2. Connect the ResNet and the SVM
    model = ResNetWithSVM(resnet, num_classes=num_classes).to(device)

    # 3. Both parts learn: the ResNet with smaller steps and its own L2 penalty
    optimizer = torch.optim.SGD([
        {"params": model.resnet.parameters(), "lr": RESNET_LR, "weight_decay": RESNET_WEIGHT_DECAY},
        {"params": model.svm.parameters(), "lr": SVM_LR},  # SVM penalty is in svm_loss
    ], momentum=0.9)

    # 4. Train, and keep the epoch with the lowest loss on test_data
    best_test_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            scores = model(images)
            loss = svm_loss(model, scores, labels)
            loss.backward()        # mistakes flow into the SVM AND the ResNet
            optimizer.step()

            running_loss += loss.item()

        test_loss, _, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{EPOCHS}  train loss: {running_loss / len(train_loader):.4f}  "
              f"test loss: {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  saved new best model (epoch {epoch + 1})")

    # 5. Final score on validate_data, which was never used for training or choosing
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
    _, predictions, y_true = evaluate(model, validate_loader, device)

    y_true = y_true.numpy()
    y_pred = predictions.numpy()
    labels = list(range(num_classes))   # every species, even ones never predicted

    print("First actual species:   ", class_names[y_true[0]])
    print("First predicted species:", class_names[y_pred[0]])
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    print("Recall:   ", recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    print("F1:       ", f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))