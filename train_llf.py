import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import Setting
from dataset import MyDataset
from model.UNet import UNetMich
from utils import save_checkpoint, load_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = r"C:\Users\hanst\Desktop\DeepLearning\PaperRevise"
TRAIN_DIR = BASE_DIR + r"\train"
VAL_DIR = BASE_DIR + r"\val"

REAL_MICH_SIZE = (155, 78)
VIRTUAL_MICH_SIZE = (155, 78)

BATCH_SIZE = 10
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
NUM_EPOCHS = 300
SAVE_FREQUENCY = 50

CHECKPOINT_DIR = "checkpoints_unet/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_NAME = "UNet-mich-noDOI-to-DOI-" + time.strftime(
    "%Y-%m-%d-%H-%M-%S", time.localtime(time.time())
)


def normalize_sum(pred, target):
    pred_sum = pred.sum((-1, -2), keepdim=True)
    target_sum = target.sum((-1, -2), keepdim=True)

    pred = pred * (target_sum / (pred_sum + 1e-8))
    pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)

    return pred


def train_one_epoch(net, loader, optimizer, criterion):
    net.train()
    total_loss = 0

    for xT, x0 in loader:
        xT = xT.to(DEVICE).float()
        x0 = x0.to(DEVICE).float()

        pred = net(xT)
        pred = normalize_sum(pred, x0)

        loss = criterion(pred, x0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()

    return total_loss / len(loader)


@torch.no_grad()
def val_one_epoch(net, loader, criterion):
    net.eval()
    total_loss = 0

    for xT, x0 in loader:
        xT = xT.to(DEVICE).float()
        x0 = x0.to(DEVICE).float()

        pred = net(xT)
        pred = normalize_sum(pred, x0)

        loss = criterion(pred, x0)
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def main():
    writer = SummaryWriter("logs/" + LOG_NAME)

    net = UNetMich(
        in_ch=1,
        out_ch=1,
        base_ch=32,
        residual=True
    ).to(DEVICE)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=[0, 1])

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0001,
        total_iters=NUM_EPOCHS
    )

    train_dataset = MyDataset(
        root_dir=TRAIN_DIR,
        real_size=REAL_MICH_SIZE,
        virtual_size=VIRTUAL_MICH_SIZE,
        total=Setting.TOTAL,
        real_area=[],
        virtural_area=[],
        virtual_mask=[]
    )

    val_dataset = MyDataset(
        root_dir=VAL_DIR,
        real_size=REAL_MICH_SIZE,
        virtual_size=VIRTUAL_MICH_SIZE,
        total=Setting.TOTAL,
        real_area=[],
        virtural_area=[],
        virtual_mask=[]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    best_val_loss = 999999

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(net, train_loader, optimizer, criterion)
        scheduler.step()

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % SAVE_FREQUENCY == 0:
            val_loss = val_one_epoch(net, val_loader, criterion)
            writer.add_scalar("val_loss", val_loss, epoch)

            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": net.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": train_loss,
                },
                CHECKPOINT_DIR + LOG_NAME + f"_epoch{epoch}.pth.tar"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state": net.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "loss": val_loss,
                    },
                    CHECKPOINT_DIR + "unet_mich_best.pth.tar"
                )

        else:
            print(f"Epoch {epoch:04d} | train_loss={train_loss:.6f}")

    writer.close()


if __name__ == "__main__":
    main()