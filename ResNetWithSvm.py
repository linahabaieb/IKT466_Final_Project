from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ResNet import Block, ResNet, dataset, train_data, test_data, validate_data
from SVM import report_metrics

# Settings
CHECKPOINT = Path(__file__).with_name("model34_20261120_113410_4")  # your trained ResNet
SAVE_PATH = Path(__file__).with_name("resnet_svm_finetuned.pt")      # best joint model
TOP_K = 20             # number of species to keep
EPOCHS = 10
BATCH_SIZE = 32
RESNET_LR = 0.001      # small nudges so the trained ResNet isn't wrecked
SVM_LR = 0.01          # bigger nudges for the new SVM layer
REGULARIZATION = 0.001 # same weight penalty as in SVM.py


def pick_top_species(subset, class_names, k):
    """Same logic as select_top_20: the k species with the most training spectrograms."""
    labels = torch.tensor([dataset.targets[i] for i in subset.indices])
    counts = torch.bincount(labels, minlength=len(class_names))
    top_ids = torch.argsort(counts, descending=True, stable=True)[:k]

    # Chosen species get new labels 0..k-1, everything else gets -1 (thrown away)
    new_label = torch.full((len(class_names),), -1, dtype=torch.long)
    for new_id, old_id in enumerate(top_ids.tolist()):
        new_label[old_id] = new_id

    top_names = [class_names[i] for i in top_ids.tolist()]
    return new_label, top_names


class RelabeledSubset(Dataset):
    """Keeps only the chosen species from a split and gives them the new labels."""
    def __init__(self, subset, new_label):
        self.base = subset.dataset
        self.new_label = new_label
        # Filter using the folder labels, so no images have to be loaded here
        self.indices = [i for i in subset.indices if new_label[self.base.targets[i]] >= 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, old_label = self.base[self.indices[i]]
        return image, self.new_label[old_label].item()


class ResNetWithSVM(nn.Module):
    """ResNet makes the 512 features, the SVM layer on top classifies them."""
    def __init__(self, resnet, num_classes):
        super().__init__()
        self.resnet = resnet
        self.resnet.fc = nn.Identity()            # remove the ResNet's own classifier
        self.svm = nn.Linear(512, num_classes)    # the SVM: one scorecard per species

    def forward(self, images):
        features = self.resnet(images)            # [batch, 512]
        return self.svm(features)                 # [batch, num_classes]


hinge_loss = nn.MultiMarginLoss(p=1, margin=1.0)  # correct species must win by ≥ 1


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

    # 1. Keep only the top 20 species in every split (chosen from the training split)
    new_label, class_names = pick_top_species(train_data, dataset.classes, TOP_K)
    train_set = RelabeledSubset(train_data, new_label)
    test_set = RelabeledSubset(test_data, new_label)
    validate_set = RelabeledSubset(validate_data, new_label)
    print(f"Train: {len(train_set)}  Test: {len(test_set)}  Validate: {len(validate_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    validate_loader = DataLoader(validate_set, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load your trained ResNet (built for all species so the weights fit)
    resnet = ResNet(Block, [3, 4, 6, 3], 3, len(dataset.classes))
    resnet.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))

    # 3. Connect the ResNet and the SVM
    model = ResNetWithSVM(resnet, num_classes=TOP_K).to(device)

    # 4. Both parts learn, the ResNet with smaller steps
    optimizer = torch.optim.SGD([
        {"params": model.resnet.parameters(), "lr": RESNET_LR},
        {"params": model.svm.parameters(), "lr": SVM_LR},
    ], momentum=0.9)

    # 5. Train, and keep the epoch with the lowest loss on test_data
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
        print(f"  train loss: {running_loss / len(train_loader):.4f}  test loss: {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print("  saved new best model")

    # 6. Final score on validate_data, which was never used for training or choosing
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
    _, predictions, y_true = evaluate(model, validate_loader, device)

    print("First actual species:   ", class_names[y_true[0].item()])
    print("First predicted species:", class_names[predictions[0].item()])
    report_metrics(y_true, predictions, class_names)