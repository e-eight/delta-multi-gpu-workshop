import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.models import resnet18
from torchvision.transforms import v2


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
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def setup():
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    acc = torch.accelerator.current_accelerator()
    backend = dist.get_default_backend_for_device(acc)
    dist.init_process_group(backend)
    return dist.get_rank(), dist.get_world_size()


def cleanup():
    dist.destroy_process_group()


def get_dataloaders(rank, world_size):
    pass


def get_model():
    pass


def reduce_metric(value, device):
    pass


def save_checkpoint(state, is_best, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "last.pt")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best.pt"))


def load_checkpoint(model, optimizer, scaler, checkpoint_path, device_id):
    if not os.path.exists(checkpoint_path):
        return 1, 0.0

    acc = torch.accelerator.current_accelerator()
    map_location = {f"{acc}:0": f"{acc}:{device_id}"}

    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    return ckpt["epoch"] + 1, ckpt.get("best_acc", 0.0)


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device):
    pass


def evaluate(model, loader, loss_fn, device):
    pass

def main():
    args = parse_args()
    rank, world_size = setup()

    train_loader, valid_loader, train_sampler = get_dataloaders(rank, world_size)

    device_id = rank % torch.accelerator.device_count()
    model = get_model().to(device_id)
    model = DDP(model, device_ids=[device_id])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

    scaler = torch.amp.GradScaler()

    writer = SummaryWriter(log_dir=args.log_dir) if rank == 0 else None

    start_epoch, best_acc = 1, 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            model.module, optimizer, scaler, args.resume, device_id
        )

    dist.barrier(device_ids=[device_id])

    epochs = 10

    for epoch in range(start_epoch, epochs + start_epoch):
        train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device_id
        )
        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn, device_id)

        train_loss = reduce_metric(train_loss, device_id)
        train_acc = reduce_metric(train_acc, device_id)
        valid_loss = reduce_metric(valid_loss, device_id)
        valid_acc = reduce_metric(valid_acc, device_id)

        if rank == 0:
            print(f"Epoch {epoch:>2}/{epochs + start_epoch}", end=" ")
            print(
                f"train_loss {train_loss:.4f} train_acc {100 * train_acc:.2f}%", end=" "
            )
            print(f"valid_loss {valid_loss:.4f} valid_acc {100 * valid_acc:.2f}%")

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", valid_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", valid_acc, epoch)

            is_best = valid_acc > best_acc
            best_acc = max(best_acc, valid_acc)

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc": best_acc,
                },
                is_best,
                args.checkpoint_dir,
            )

        dist.barrier(device_ids=[device_id])

    cleanup()


if __name__ == "__main__":
    main()
