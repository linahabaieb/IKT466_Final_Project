from pathlib import Path
import torch
from ResNet import Block, ResNet, dataset, train_data, validate_data
from SVM import get_features, train_svm, report_metrics, select_top_20

# Loading the trained ResNet
checkpoint = Path(__file__).with_name("model34_20261120_113410_4")
resnet = ResNet(Block, [3, 4, 6, 3], 3, len(dataset.classes))
weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
resnet.load_state_dict(weights)
resnet.eval()

# Returning the 512 features that normally go into the final classifier.
resnet.fc = torch.nn.Identity()

feature_folder = checkpoint.parent / "features"
feature_folder.mkdir(exist_ok=True)
feature_file = feature_folder / (checkpoint.name + ".pt")

if feature_file.exists():
    print("Loading saved features...")
    X_train, y_train, X_final, y_final = torch.load(
        feature_file, map_location="cpu", weights_only=True
    )
else:
    print("Extracting training features...")
    X_train, y_train = get_features(resnet, train_data)

    print("Extracting evaluation features...")
    X_final, y_final = get_features(resnet, validate_data)

    torch.save((X_train, y_train, X_final, y_final), feature_file)
    print("Saved features for next time.")

use_top_20 = True
class_names = dataset.classes

if use_top_20:
    X_train, y_train, X_final, y_final, class_names = select_top_20(
        X_train, y_train, X_final, y_final, class_names
    )

print("Number of species in training:", y_train.unique().numel())
print("Training feature table:", X_train.shape)
print("Evaluation feature table:", X_final.shape)

svm = train_svm(X_train, y_train, num_classes=len(class_names))

with torch.no_grad():
    scores = svm(X_final)
    predictions = scores.argmax(dim=1)

print("First actual species:", class_names[y_final[0].item()])
print("First predicted species:", class_names[predictions[0].item()])

report_metrics(y_final, predictions, class_names)