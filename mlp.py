import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.hidden = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, num_classes)

    def forward(self, features):
        x = self.hidden(features)
        x = self.relu(x)
        scores = self.output(x)
        return scores


def train_mlp(x_train, y_train, x_val, y_val, num_classes, epochs=30):
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

    model = MLP(x_train.shape[1], num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            loss = loss_fn(model(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = model(x_val.to(device))
            val_loss = nn.functional.cross_entropy(scores, y_val.to(device)).item()
            val_acc = (scores.argmax(1) == y_val.to(device)).float().mean().item()
        print(f'Epoch {epoch + 1}: validation loss {val_loss:.4f}, accuracy {val_acc:.2%}')

    return model


if __name__ == '__main__':
    x_train, y_train = torch.randn(500, 512), torch.randint(0, 10, (500,))
    x_val, y_val = torch.randn(100, 512), torch.randint(0, 10, (100,))
    train_mlp(x_train, y_train, x_val, y_val, num_classes=10, epochs=3)