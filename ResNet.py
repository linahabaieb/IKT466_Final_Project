import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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
from sklearn.metrics import precision_score, recall_score, accuracy_score

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(SCRIPT_PATH,"spectrograms")

dataset = datasets.ImageFolder(data_dir,transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))

class_sizes = Counter(dataset.targets)

usable_classes = {
    class_id for class_id, count in class_sizes.items() if count > 10
}

usable_indices = [
    data_index for data_index, target in enumerate(dataset.targets) if target in usable_classes
]

filtered_labels = [dataset.targets[i] for i in usable_indices]

train_indexes, temp_indexes, train_labels, temp_labels = train_test_split(usable_indices, filtered_labels,train_size=0.7,random_state=42,stratify=filtered_labels)
test_indexes, validate_indexes = train_test_split(temp_indexes,train_size=0.67,random_state=42,stratify=temp_labels)

train_data = Subset(dataset, train_indexes)
test_data = Subset(dataset, test_indexes)
validate_data = Subset(dataset, validate_indexes)

loaded_training_data = DataLoader(train_data,batch_size=64,shuffle=True)
loaded_test_data = DataLoader(test_data,batch_size=64,shuffle=True)
loaded_validation_data = DataLoader(validate_data,batch_size=64,shuffle=True)



class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        super(Block,self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.batch_norm_layer1 = nn.BatchNorm2d(out_channels)
        self.conv_layer2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.batch_norm_layer2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self,x):
        identity = x

        x = self.conv_layer1(x)
        x = self.batch_norm_layer1(x)
        x = self.conv_layer2(x)
        x = self.batch_norm_layer2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        activated = self.relu(x)
        return activated

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes): 
        super(ResNet,self).__init__()

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

        final_channels = 512

        self.input_shortcut = nn.Sequential(
            nn.Conv2d(image_channels, final_channels, kernel_size=1, stride=32, bias=False),
            nn.BatchNorm2d(final_channels),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, layer_size, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,out_channels, kernel_size=1, stride = stride), nn.BatchNorm2d(out_channels))

        layers.append(block(self.in_channels,out_channels, identity_downsample, stride))
        self.in_channels = out_channels

        for i in range(layer_size-1):
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)

    def forward(self,x):
        identity = self.input_shortcut(x)
        x = self.conv1(x)
        x = self.batch_norm_layer1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x + identity

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def train_epoch(batch_report=20):
    total_loss = 0
    last_loss = 0

    for index, data in enumerate(loaded_training_data):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = resNet34(inputs)

        loss = loss_fn(outputs,labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        if (index + 1) % batch_report == 0:
            last_loss = total_loss / batch_report
            print(f'  batch {index + 1} loss: {last_loss}')
            total_loss = 0.
    return last_loss

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resNet34 = ResNet(Block,[3,4,6,3],3,len(dataset.classes)).to(device)
    print("Using:", device)
    if args.train == True:
        
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(resNet34.parameters(), lr=0.01, momentum=0.9)

        current_epoch = 0

        total_epochs = 5

        best_loss = 1_000_000

        for epoch in range(total_epochs):
            print(f"EPOCH: {current_epoch}")

            resNet34.train(True)
            avg_loss = train_epoch()


            total_test_loss = 0
            resNet34.eval()

            with torch.no_grad():
                for test_index, test_data in enumerate(loaded_test_data):
                    test_inputs, test_labels = test_data
                    test_inputs = test_inputs.to(device)
                    test_labels = test_labels.to(device)
                    test_outputs = resNet34(test_inputs)
                    test_loss = loss_fn(test_outputs,test_labels)

                    total_test_loss += test_loss.item()

            avg_test_loss = total_test_loss/(len(loaded_test_data)+1)

            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                model_path = f"model34_{timestamp}_{current_epoch}"
                torch.save(resNet34.state_dict(), model_path)

            current_epoch += 1

    matches = glob(f"model34_*_4")

    if not matches:
        raise FileNotFoundError(f"No checkpoints found for {timestamp}")

    latest_day = max(matches, key=lambda path: int(os.path.basename(path).split("_")[-3]))
    tied_for_latest_day = [state_file for state_file in matches if int(os.path.basename(state_file).split("_")[-3]) == int(os.path.basename(latest_day).split("_")[-3])]
    latest_time = max(tied_for_latest_day, key=lambda path: int(os.path.basename(path).split("_")[-2]))
    tied_for_latest_time = [state_file for state_file in tied_for_latest_day if int(os.path.basename(state_file).split("_")[-3]) == int(os.path.basename(latest_time).split("_")[-3])]
    best_state = max(tied_for_latest_time, key=lambda path: int(os.path.basename(path).split("_")[-1]))

    state_dict = torch.load(best_state, map_location=device, weights_only=True)
    resNet34.load_state_dict(state_dict)
    resNet34.eval()

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, true_labels in loaded_validation_data:
            images = images.to(device)
            predictions = resNet34(images).argmax(dim=1).cpu()

            for true_label, prediction in zip(true_labels,predictions):
                all_predictions.append(prediction)
                all_targets.append(true_label)

    topX = 10
    sorted_class_sizes = sorted(class_sizes.items(), key=lambda pair: pair[1],reverse=True)
    top_classes = sorted_class_sizes[:topX]

    top_class_ids = {label for label, _ in top_classes}
    top_classses_predicitons = [(pred.item(),target.item()) for pred, target in zip(all_predictions,all_targets) if target.item() in top_class_ids]

    sortet_predictions = [t[0] for t in top_classses_predicitons]
    sortet_targets = [t[1] for t in top_classses_predicitons]
    sk_accuracy = accuracy_score(sortet_targets, sortet_predictions)
    sk_precision = precision_score(sortet_targets, sortet_predictions,average="macro", zero_division=0)
    sk_recall = recall_score(sortet_targets, sortet_predictions,average="macro", zero_division=0)
    print(f"Top Overall sk_accuracy:{sk_accuracy}")
    print(f"Top Overall sk_precision: {sk_precision}")
    print(f"Top Overall sk_recall: {sk_recall}")

    per_class_predictions = {}
    per_class_scores = {}
    for int_label in dataset.class_to_idx.values():
        class_results = [t for t in top_classses_predicitons if  int(int_label) == int(t[1])]
        if len(class_results) == 0:
            continue
        classs_predictions = [t[0] for t in class_results]
        class_targets = [t[1] for t in class_results]
        per_class_predictions[int_label] = [classs_predictions,class_targets]
        class_acc = accuracy_score(class_targets, classs_predictions)
        class_prec = precision_score(class_targets, classs_predictions,average="macro", zero_division=0)
        class_rec = recall_score(class_targets, classs_predictions,average="macro", zero_division=0)
        per_class_scores[int_label] = (class_acc,class_prec,class_rec)

    print(f"| Class | Accuracy | Precision | Recall |")
    for key, value in per_class_scores.items():
        print(f"|{key}|{round(value[0],2)}|{round(value[1],2)}|{round(value[2],2)}|")

    print(f"Top {topX} classes: {top_classes}")


    ##########################################################################


    label_counter = dataset.class_to_idx
    counter_per_label = {int_label:Counter() for int_label in label_counter.values()}

    with torch.no_grad():
        for i in tqdm(range(min(2000, len(validate_data)))):
            image, true_label = validate_data[i]
            with torch.no_grad():
                image = image.unsqueeze(0).to(device)
                logits = resNet34(image)
                predicted_id = logits.argmax(dim=1).item()

            prediction_bool = predicted_id == true_label

            counter_per_label[true_label][prediction_bool] += 1

    eval_subset_targets = [validate_data[i][1] for i in range(len(validate_data))]
    eval_class_sizes = Counter(eval_subset_targets)

    accuracy_per_label = []
    accuracy_sum = 0
    for label, counter in counter_per_label.items():
        total = counter[True] + counter[False]
        accuracy = counter[True] / total if total else 0.0
        accuracy_per_label.append((label,accuracy,class_sizes[label]))
        accuracy_sum += accuracy
    accuracy_avg = accuracy_sum/len(counter_per_label)
    print(f"Accuracy Per Class: {accuracy_per_label}")






    