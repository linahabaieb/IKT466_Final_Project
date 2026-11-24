import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def get_features(model, data):
    features = []
    labels = []

    with torch.no_grad():
        for images, species in DataLoader(data, batch_size=32):
            features.append(model(images).cpu())
            labels.append(species.cpu())

    return torch.cat(features), torch.cat(labels)

def train_svm(X_train, y_train, num_classes):
    seed = 42
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_data = TensorDataset(X_train, y_train)
    loader = DataLoader(train_data, batch_size=64, shuffle=True, generator=generator)

    svm = nn.Linear(X_train.shape[1], num_classes)
    #Creates the rule for measuring prediction errors during training
    hinge_loss = nn.MultiMarginLoss(p=1, margin=1.0)
    optimizer = torch.optim.SGD(svm.parameters(), lr=0.01)
    regularization = 0.001

    epochs = 50
    print(f"Total epochs: {epochs}")
    for epoch in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad()

            scores = svm(features)
            data_loss = hinge_loss(scores, labels)
            weight_penalty = svm.weight.square().sum()
            loss = data_loss + regularization * weight_penalty

            loss.backward()
            optimizer.step()
            
    return svm

def report_metrics(y_true, predictions, class_names):
    accuracy = (predictions == y_true).float().mean().item()
    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for species_id in torch.unique(torch.cat([y_true, predictions])):
        actual = y_true == species_id
        guessed = predictions == species_id

        correct = (actual & guessed).sum().item()
        support = actual.sum().item()
        total_guesses = guessed.sum().item()

        precision = correct / total_guesses if total_guesses else 0.0
        recall = correct / support if support else 0.0
        f1 = 2 * correct / (support + total_guesses)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"SVM accuracy:  {accuracy:.1%}")
    print(f"SVM precision: {sum(precision_scores) / len(precision_scores):.1%}")
    print(f"SVM recall:    {sum(recall_scores) / len(recall_scores):.1%}")
    print(f"SVM F1:        {sum(f1_scores) / len(f1_scores):.1%}")
    
def select_top_20(X_train, y_train, X_final, y_final, class_names):
    # Count how many training spectrograms belong to each species.
    counts = torch.bincount(y_train, minlength=len(class_names))

    # Sort from most to fewest spectrograms and take the first 20 species IDs.
    top_ids = torch.argsort(counts, descending=True, stable=True)[:20]
    top_names = [class_names[species_id] for species_id in top_ids.tolist()]

    # Give the selected species new labels from 0 to 19.
    # All other species get -1 so we can filter them out.
    new_label = torch.full((len(class_names),), -1, dtype=torch.long)
    for new_id, old_id in enumerate(top_ids.tolist()):
        new_label[old_id] = new_id

    train_labels = new_label[y_train]
    final_labels = new_label[y_final]

    keep_train = train_labels >= 0
    keep_final = final_labels >= 0
    
    return (
    X_train[keep_train], train_labels[keep_train],
    X_final[keep_final], final_labels[keep_final], 
    top_names,
    )