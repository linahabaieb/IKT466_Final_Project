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

from torchvision.models import resnet50, ResNet50_Weights

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(SCRIPT_PATH,"spectrograms")

dataset = datasets.ImageFolder(data_dir,transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )]))

counts = Counter(dataset.targets)

usable_classes = {
    class_id for class_id, count in counts.items() if count > 10
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

loaded_training_data = DataLoader(train_data,batch_size=32,shuffle=True)
loaded_test_data = DataLoader(test_data,batch_size=32,shuffle=True)
loaded_validation_data = DataLoader(validate_data,batch_size=32,shuffle=True)


weights = ResNet50_Weights.DEFAULT
model  = resnet50(weights=weights)

num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

parser = argparse.ArgumentParser()
parser.add_argument("--predict", action="store_true")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model = model.to(device=device)

def train_epoch(batch_report=20):
    total_loss = 0
    last_loss = 0

    for index, data in enumerate(loaded_training_data):
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

if args.predict == True:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    current_epoch = 0

    total_epochs = 5

    best_loss = 1_000_000

    for epoch in range(total_epochs):
        print(f"EPOCH: {current_epoch}")

        model.train(True)
        avg_loss = train_epoch()


        total_test_loss = 0
        model.eval()

        with torch.no_grad():
            for test_index, test_data in enumerate(loaded_test_data):
                test_inputs, test_labels = test_data
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)
                test_outputs = model(test_inputs)
                test_loss = loss_fn(test_outputs,test_labels)

                total_test_loss += test_loss.item()

        avg_test_loss = total_test_loss/len(loaded_test_data)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            model_path = f"model50_pretrained_{timestamp}_{current_epoch}"
            torch.save(model.state_dict(), model_path)

        current_epoch += 1

matches = glob(f"model34_{timestamp}_*")

if not matches:
    raise FileNotFoundError(f"No checkpoints found for {timestamp}")

best_state = max(matches, key=lambda path: int(os.path.basename(path).rsplit("_", 1)[-1]))

state_dict = torch.load(best_state, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

accuracy_counter = Counter()
for i in tqdm(range(min(2000, len(validate_data)))):
    image, true_label = validate_data[i]
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        logits = model(image)
        predicted_id = logits.argmax(dim=1).item()

    prediction_truth = dataset.classes[predicted_id] == dataset.classes[true_label]
    accuracy_counter[prediction_truth] += 1
print(f"Accuracy: {accuracy_counter[True]/(accuracy_counter[True]+accuracy_counter[False])}")
