import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
from  torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from datetime import datetime
import argparse
from tqdm import tqdm
from glob import glob
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--top_20", action="store_true")

args = parser.parse_args()

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(SCRIPT_PATH,"spectrograms")

dataset = datasets.ImageFolder(data_dir,transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))

class_sizes = Counter(dataset.targets)

if args.top_20:
    top_20_class_ids = {
        class_id
        for class_id, count in class_sizes.most_common(20)
    }

    usable_class_ids = sorted(top_20_class_ids)
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

loaded_training_data = DataLoader(train_data, batch_size=64, shuffle=True)
loaded_test_data = DataLoader(test_data, batch_size=64, shuffle=False)
loaded_validation_data = DataLoader(validate_data, batch_size=64, shuffle=False)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        super(Block,self).__init__()
        self.expansion = 4
        self.conv_layer1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.batch_norm_layer1 = nn.BatchNorm2d(out_channels)
        self.conv_layer2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.batch_norm_layer2 = nn.BatchNorm2d(out_channels)
        self.conv_layer3 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=3,stride=1,padding=1)
        self.batch_norm_layer3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self,x):
        identity = x

        x = self.conv_layer1(x)
        x = self.batch_norm_layer1(x)
        x = self.conv_layer2(x)
        x = self.batch_norm_layer2(x)
        x = self.conv_layer3(x)
        x = self.batch_norm_layer3(x)


        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        activated = self.relu(x)
        return activated

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes): 
        super(ResNet,self).__init__()
        self.expansion = 4
        # Layers
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels,64,kernel_size=7,stride=2,padding=3)
        self.batch_norm_layer1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(block,layers[0],64,1)
        self.layer2 = self._make_layer(block,layers[1],128,2)
        self.layer3 = self._make_layer(block,layers[2],256,2)
        self.layer4 = self._make_layer(block,layers[3],512,2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(self, block, layer_size, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels*self.expansion:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,out_channels*self.expansion, kernel_size=1, stride = stride), nn.BatchNorm2d(out_channels*self.expansion))

        layers.append(block(self.in_channels,out_channels, identity_downsample, stride))
        self.in_channels = out_channels*self.expansion

        for i in range(layer_size-1):
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)

    def forward_features(self, x):

        x = self.conv1(x)
        x = self.batch_norm_layer1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        return torch.flatten(x, 1)

    def forward(self,x):
        features = self.forward_features(x)
        return self.fc(features)


def ResNet50(num_classes,img_channels=3):
    return ResNet(Block,[3,4,6,3],img_channels,num_classes)

def train_epoch(model, data_loader, optimizer, loss_fn, device, batch_report=20):
    model.train()
    total_loss = 0
    last_loss = 0

    if args.top_20 == True:
        batch_report = 2

    for index, data in enumerate(data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs,labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        if (index + 1) % batch_report == 0:
            last_loss = total_loss / batch_report
            print(f'  batch {index + 1} loss: {last_loss}')
            total_loss = 0.
    return last_loss

def train(model, time_stamp, train_dataloader, test_dataloader, device, loss_function):
    loss_fn = loss_function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    current_epoch = 0

    total_epochs = 5

    best_loss = float('inf')

    for _ in range(total_epochs):
        print(f"EPOCH: {current_epoch}")

        model.train(True)
        _ = train_epoch(model,train_dataloader,optimizer,loss_fn,device)

        total_test_loss = 0
        model.eval()

        with torch.no_grad():
            for test_inputs, test_labels in test_dataloader:
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)

                test_outputs = model(test_inputs)
                test_loss = loss_fn(test_outputs, test_labels)
                total_test_loss += test_loss.item()

        avg_test_loss = total_test_loss / len(test_dataloader)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            model_path = f"model50_{time_stamp}_{current_epoch}"
            torch.save(model.state_dict(), model_path)

        current_epoch += 1

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # SELECT DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # MODEL
    net50 = ResNet50(num_classes).to(device)

    # TRAINING
    if args.train == True:
        train_label_counts = torch.bincount(
            torch.tensor(train_labels, dtype=torch.long),
            minlength=num_classes,
        ).float()

        class_weights = (
            train_label_counts.sum()
            / (num_classes * train_label_counts.clamp_min(1))
        ).to(device)
        
        loss_fn = nn.CrossEntropyLoss(weight=None)
        train(net50,timestamp, loaded_training_data, loaded_test_data, device, loss_fn)

    # LOADING STATE
    matches = glob(f"model50_*")

    if not matches:
        raise FileNotFoundError(f"No checkpoints found for {timestamp}")

    latest_day = max(matches, key=lambda path: int(os.path.basename(path).split("_")[-3]))
    tied_for_latest_day = [state_file for state_file in matches if int(os.path.basename(state_file).split("_")[-3]) == int(os.path.basename(latest_day).split("_")[-3])]
    latest_time = max(tied_for_latest_day, key=lambda path: int(os.path.basename(path).split("_")[-2]))
    tied_for_latest_time = [state_file for state_file in tied_for_latest_day if int(os.path.basename(state_file).split("_")[-2]) == int(os.path.basename(latest_time).split("_")[-2])]
    best_state = max(tied_for_latest_time, key=lambda path: int(os.path.basename(path).split("_")[-1]))

    #best_state = "model34_20261124_105946_4"
    print(best_state)
    state_dict = torch.load(best_state, map_location=device, weights_only=True)
    net50.load_state_dict(state_dict)
    net50.eval()

    # SCORING 
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, true_labels in loaded_validation_data:
            images = images.to(device)
            predictions = net50(images).argmax(dim=1).cpu()

            for true_label, prediction in zip(true_labels,predictions):
                all_predictions.append(prediction)
                all_targets.append(true_label)

    y_true = torch.stack(all_targets).numpy()
    y_pred = torch.stack(all_predictions).numpy()
    labels = list(range(num_classes))

    print("Validation accuracy:", accuracy_score(y_true, y_pred))

    top_20_classes_og_ids = {
        class_id for class_id, count in class_sizes.most_common(20)
    }

    top_20_classes_new_ids = {
        old_to_new[old_id] for old_id in top_20_classes_og_ids if old_id in old_to_new
    }

    mask = np.isin(y_true,list(top_20_classes_new_ids))

    if mask.any():
        print("Top 20 Validation accuracy:", accuracy_score(y_true[mask], y_pred[mask]))
        print("Top 20 Validation F1:", f1_score(y_true[mask], y_pred[mask], average="macro", labels=labels, zero_division=0))
        print("Top 20 Validation Recall:", recall_score(y_true[mask], y_pred[mask], average="macro",labels=labels, zero_division=0))
    else:
        print("No class in the top 20 was in the validation examples. R.I.P")

    print(
        "Validation macro-F1:",
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    )
    print(
        "Validation macro-recall:",
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    )

    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    for class_id, recall in enumerate(per_class_recall):
        print(f"ID:{class_id} {class_names[class_id]} recall: {recall:.3f}")

    # GRAD CAM
    net50.eval()

    target_layers = [net50.layer4[-1].conv_layer2]
    image, true_label = validate_data[24]
    input_tensor = image.unsqueeze(0).to(device)

    target = int(true_label)

    with GradCAM(net50,target_layers) as cam:
        targets = [ClassifierOutputTarget(target)]

        grayscale_cam = cam(input_tensor,targets)
        grayscale_cam = grayscale_cam[0, :]

    rgb_image = image.permute(1, 2, 0).cpu().numpy()
    rgb_image = np.clip(rgb_image, 0, 1)
    
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)
    plt.title(f"Input — {class_names[target]}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
