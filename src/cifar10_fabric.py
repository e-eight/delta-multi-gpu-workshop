import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.models import resnet18
from torchvision.transforms import v2
from wandb.integration.lightning.fabric import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Image Classification with DDP")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        metavar="DIR",
        help="Directory to download the data to",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        metavar="DIR",
        help="Directory to save logs to",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        metavar="DIR",
        help="Directory to save checkpoints to",
    )
    parser.add_argument(
        "--save-checkpoint-interval",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to save model after",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def get_loaders(fabric, data_dir):
    cifar10_normalization = v2.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    train_transform = v2.Compose(
        [
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            cifar10_normalization,
        ]
    )
    valid_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            cifar10_normalization,
        ]
    )

    with fabric.rank_zero_first(local=False):
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=fabric.is_global_zero,
            transform=train_transform,
        )
        valid_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=fabric.is_global_zero,
            transform=valid_transform,
        )

    train_loader = DataLoader(train_dataset, batch_size=64)
    valid_loader = DataLoader(valid_dataset, batch_size=64)

    return fabric.setup_dataloaders(train_loader, valid_loader)


def get_model_and_optimizer(fabric):
    model = resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, out_features=10, bias=True)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    return fabric.setup(model, optimizer)


def train_one_epoch(model, loader, loss_fn, optimizer, fabric, metric):
    running_loss = 0.0
    metric.reset()
    model.train()
    for inputs, targets in loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        fabric.backward(loss)
        optimizer.step()

        running_loss += loss.item()
        metric.update(outputs, targets)

    running_loss = fabric.all_gather(running_loss).sum() / len(loader.dataset)
    return running_loss, metric.compute()


def evaluate(model, loader, loss_fn, fabric, metric):
    running_loss = 0.0
    metric.reset()
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            running_loss += loss_fn(outputs, targets).item()
            metric.update(outputs, targets)
    running_loss = fabric.all_gather(running_loss).sum() / len(loader.dataset)
    return running_loss, metric.compute()


def main():
    args = parse_args()
    tb_logger = TensorBoardLogger(root_dir=args.log_dir, name="multi-gpu-cifar10")
    wb_logger = WandbLogger(project="multi-gpu-cifar10")

    fabric = Fabric(loggers=[tb_logger, wb_logger])
    seed_everything(42)

    train_loader, valid_loader = get_loaders(fabric, args.data_dir)
    model, optimizer = get_model_and_optimizer(fabric)
    loss_fn = nn.CrossEntropyLoss()

    accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    start_epoch = 1
    if args.resume:
        state = {"model": model, "optimizer": optimizer, "epoch": start_epoch}

    epochs = 10
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, fabric, accuracy
        )
        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn, fabric, accuracy)

        if fabric.is_global_zero:
            fabric.print(f"epoch {epoch:>2d}", end=" ")
            fabric.print(
                f"train loss: {train_loss:>.3e}, train acc: {100 * train_acc:>5.1f}%",
                end=" ",
            )
            fabric.print(
                f"valid loss: {valid_loss:>.3e}, valid acc: {100 * valid_acc:>5.1f}%",
            )

        metrics = {
            "train_loss": train_loss,
            "train_acc": 100 * train_acc,
            "valid_loss": valid_loss,
            "valid_acc": 100 * valid_acc,
        }
        fabric.log_dict(metrics, epoch)

        if epoch % args.save_checkpoint_interval == 0:
            state = {"model": model, "optimizer": optimizer, "epoch": epoch}
            fabric.save(os.path.join("checkpoints", f"checkpoint_{epoch}.ckpt"), state)


if __name__ == "__main__":
    main()
