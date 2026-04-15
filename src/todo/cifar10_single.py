import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.models import resnet18


train_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
valid_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
valid_dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=1)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    pass


def evaluate(model, loader, loss_fn, device):
    pass

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 10, bias=True)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 10

for epoch in range(1, epochs + 1): 
    train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn, device)
    print(f"Epoch {epoch:>2}/{epochs}", end=" ")
    print(f"train_loss {train_loss:.4f} train_acc {100 * train_acc:.2f}%", end=" ")
    print(f"valid_loss {valid_loss:.4f} valid_acc {100 * valid_acc:.2f}%")
