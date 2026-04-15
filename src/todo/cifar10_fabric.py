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

def get_loaders(fabric):
    pass


def get_model_and_optimizer(fabric):
    pass

def train_one_epoch(model, loader, loss_fn, optimizer, fabric, metric):
    pass


def evaluate(model, loader, loss_fn, fabric, metric):
    pass


def main():
    args = parse_args()
    tb_logger = TensorBoardLogger(root_dir=args.log_dir, name="multi-gpu-cifar10")
    wb_logger = WandbLogger(project="multi-gpu-cifar10")

    fabric = Fabric(loggers=[tb_logger, wb_logger])
    seed_everything(42)

    train_loader, valid_loader = get_loaders(fabric)
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

        if epoch % args.save_checkout_interval == 0:
            state = {"model": model, "optimizer": optimizer, "epoch": epoch}
            fabric.save(os.path.join("checkpoints", f"checkpoint_{epoch}.ckpt"), state)


if __name__ == "__main__":
    main()
