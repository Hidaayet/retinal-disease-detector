"""
Evaluation script — prints per-class metrics and saves a confusion matrix.

Usage:
    python model/evaluate.py \
        --data_dir   data/ \
        --model_path data/best_model.pth \
        --output_dir outputs/
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RetinalDataset, get_transforms
from model import RetinalClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all, probs_all = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds_all.extend(torch.argmax(outputs, 1).cpu().numpy())
            labels_all.extend(labels.numpy())
            probs_all.append(probs.cpu().numpy())

    return (
        np.array(labels_all),
        np.array(preds_all),
        np.vstack(probs_all),
    )


def main(args):
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ────────────────────────────────────────────────────────────────
    df = pd.read_csv(f"{args.data_dir}/train.csv")
    df["filepath"] = df["id_code"].apply(
        lambda x: f"{args.data_dir}/train_images/{x}.png"
    )
    _, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["diagnosis"]
    )

    val_loader = DataLoader(
        RetinalDataset(val_df, transform=get_transforms(is_train=False)),
        batch_size=16, shuffle=False, num_workers=0,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = RetinalClassifier(num_classes=5).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logger.info("Loaded model from %s", args.model_path)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    labels, preds, probs = evaluate(model, val_loader, device)

    kappa = cohen_kappa_score(labels, preds, weights="quadratic")
    accuracy = accuracy_score(labels, preds)
    logger.info("Quadratic Weighted Kappa : %.4f", kappa)
    logger.info("Accuracy                 : %.4f", accuracy)
    print("\n" + classification_report(labels, preds, target_names=GRADE_NAMES))

    # ── Confidence analysis ──────────────────────────────────────────────────
    max_probs = probs.max(axis=1)
    low_conf = (max_probs < 0.6).sum()
    logger.info(
        "Low-confidence predictions (<60%%): %d / %d (%.1f%%)",
        low_conf, len(labels), 100 * low_conf / len(labels),
    )

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GRADE_NAMES, yticklabels=GRADE_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix  (κ = {kappa:.4f})")
    plt.tight_layout()
    out_path = f"{args.output_dir}/confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    logger.info("Confusion matrix saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retinal disease classifier")
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--model_path", default="data/best_model.pth")
    parser.add_argument("--output_dir", default="outputs")
    main(parser.parse_args())