"""
Training script for the retinal disease detector.

Usage:
    python model/train.py \
        --data_dir  data/ \
        --save_path data/best_model.pth \
        --epochs    10 \
        --batch_size 16 \
        --lr        1e-4
"""

import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RetinalDataset, get_transforms
from model import RetinalClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, preds_all, labels_all = 0.0, [], []

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds_all.extend(torch.argmax(outputs, 1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = accuracy_score(labels_all, preds_all)
    kappa = cohen_kappa_score(labels_all, preds_all, weights="quadratic")
    return avg_loss, accuracy, kappa


def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, preds_all, labels_all = 0.0, [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val  ", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds_all.extend(torch.argmax(outputs, 1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = accuracy_score(labels_all, preds_all)
    kappa = cohen_kappa_score(labels_all, preds_all, weights="quadratic")
    return avg_loss, accuracy, kappa


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Data ────────────────────────────────────────────────────────────────
    df = pd.read_csv(f"{args.data_dir}/train.csv")
    df["filepath"] = df["id_code"].apply(
        lambda x: f"{args.data_dir}/train_images/{x}.png"
    )

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["diagnosis"]
    )
    logger.info("Train: %d  |  Val: %d", len(train_df), len(val_df))

    train_loader = DataLoader(
        RetinalDataset(train_df, transform=get_transforms(is_train=True)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        RetinalDataset(val_df, transform=get_transforms(is_train=False)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # ── Class weights (inverse frequency) ───────────────────────────────────
    counts = train_df["diagnosis"].value_counts().sort_index().values
    weights = torch.FloatTensor((1.0 / counts) / (1.0 / counts).sum() * 5).to(device)
    logger.info("Class weights: %s", weights.tolist())

    # ── Model / loss / optimiser ─────────────────────────────────────────────
    model = RetinalClassifier(num_classes=5).to(device)
    total, trainable = model.count_parameters()
    logger.info("Parameters — total: %s  trainable: %s",
                f"{total:,}", f"{trainable:,}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_kappa = 0.0
    header = f"{'Epoch':>5} {'T-Loss':>8} {'T-Acc':>7} {'T-κ':>7} " \
             f"{'V-Loss':>8} {'V-Acc':>7} {'V-κ':>7}"
    logger.info(header)

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc, t_kappa = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_kappa = val_epoch(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        flag = " ← best" if v_kappa > best_kappa else ""
        logger.info(
            "%5d %8.4f %7.4f %7.4f %8.4f %7.4f %7.4f%s",
            epoch, t_loss, t_acc, t_kappa, v_loss, v_acc, v_kappa, flag,
        )
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        if v_kappa > best_kappa:
            best_kappa = v_kappa
            torch.save(model.state_dict(), args.save_path)

    logger.info("Training complete. Best val κ: %.4f", best_kappa)
    logger.info("Model saved to: %s", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train retinal disease classifier")
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--save_path",  default="data/best_model.pth")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    main(parser.parse_args())