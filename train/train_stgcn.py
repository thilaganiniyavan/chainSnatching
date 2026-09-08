"""Training Script for ST-GCN Skeleton Action Recognition Model.

Trains Spatial-Temporal Graph Convolutional Network on CCTV skeleton sequences with:
- Spatial graph convolution on normalized COCO-17 skeletal topology.
- Temporal convolution with residual connections.
- Class-weighted CrossEntropyLoss with Label Smoothing (0.1).
- CosineAnnealingWarmRestarts learning rate schedule.
- Checkpoint export to models/stgcn_coco17.pth.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.action.stgcn_recognizer import (
    STGCNModelPyTorch,
    _COCO_EDGES,
)
from src.action.base_recognizer import DEFAULT_ACTION_TAXONOMY as ACTION_TAXONOMY
from train.dataset import SkeletonActionDataset


def build_normalized_adjacency(num_joints: int = 17) -> torch.Tensor:
    """Build symmetrically normalized adjacency matrix D^(-0.5) (A + I) D^(-0.5)."""
    adj_np = np.eye(num_joints, dtype=np.float32)
    for i, j in _COCO_EDGES:
        if i < num_joints and j < num_joints:
            adj_np[i, j] = 1.0
            adj_np[j, i] = 1.0

    deg = np.sum(adj_np, axis=1)
    deg_inv = np.power(deg, -0.5, where=deg > 0)
    deg_inv[deg == 0] = 0.0
    norm_adj = np.diag(deg_inv) @ adj_np @ np.diag(deg_inv)
    return torch.tensor(norm_adj, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    adj: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Execute one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch, adj)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    adj: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, dict[int, float]]:
    """Evaluate model performance on validation split."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch, adj)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

            for p, y in zip(preds.cpu().numpy(), y_batch.cpu().numpy()):
                class_correct[y] = class_correct.get(y, 0) + int(p == y)
                class_total[y] = class_total.get(y, 0) + 1

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    per_class_acc = {c: class_correct.get(c, 0) / max(1, class_total.get(c, 1)) for c in class_total}
    return avg_loss, acc, per_class_acc


def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN Skeleton Action Model")
    parser.add_argument("--data", type=str, default="data/skeleton_dataset.npz", help="Dataset path")
    parser.add_argument("--output", type=str, default="models/stgcn_coco17.pth", help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Initial learning rate")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Dataset file not found at {args.data}. Run train/extract_skeleton_dataset.py first.")
        return

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f" ST-GCN ACTION RECOGNITION TRAINING")
    print(f" Device: {device} | Checkpoint Target: {args.output}")
    print("=" * 80)

    # Load data and build train/validation split (80/20)
    raw = np.load(args.data, allow_pickle=True)
    N = len(raw["y"])
    np.random.seed(42)
    indices = np.random.permutation(N)
    split_idx = int(0.80 * N)
    train_indices = indices[:split_idx].tolist()
    val_indices = indices[split_idx:].tolist()

    train_ds = SkeletonActionDataset(args.data, augment=True, indices=train_indices)
    val_ds = SkeletonActionDataset(args.data, augment=False, indices=val_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f" Total Samples: {N} (Train: {len(train_ds)}, Val: {len(val_ds)})")

    # Compute smoothed class weights for loss balancing (square-root inverse frequency)
    y_train = raw["y"][train_indices]
    counts = np.bincount(y_train, minlength=len(ACTION_TAXONOMY))
    weights = np.where(counts > 0, 1.0 / np.power(np.maximum(counts, 1), 0.5), 0.0)
    weights = weights / (np.sum(weights) + 1e-6) * len(ACTION_TAXONOMY)
    class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    # Instantiate model
    model = STGCNModelPyTorch(
        in_channels=4,
        num_classes=len(ACTION_TAXONOMY),
        num_joints=17,
    ).to(device)

    adj = build_normalized_adjacency(num_joints=17).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.10)

    best_val_acc = 0.0
    start_time = time.time()

    print("\nStarting Training Loop:")
    print(f"{'Epoch':<8} | {'Train Loss':<11} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Status'}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, adj, device)
        val_loss, val_acc, per_class = evaluate(model, val_loader, criterion, adj, device)
        scheduler.step()

        is_best = val_acc > best_val_acc
        status = "[BEST]" if is_best else ""
        if is_best:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "taxonomy": ACTION_TAXONOMY,
                "epoch": epoch,
                "val_acc": val_acc,
                "in_channels": 4,
                "num_joints": 17,
            }, args.output)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"{epoch:03d}/{args.epochs:03d}  | {tr_loss:<11.4f} | {tr_acc * 100:<9.1f}% | {val_loss:<10.4f} | {val_acc * 100:<9.1f}% | {status}")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"\n Training finished in {elapsed:.1f}s ({elapsed / 60:.2f} min).")
    print(f" Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f" Model saved to: {os.path.abspath(args.output)} ({os.path.getsize(args.output) / 1024:.1f} KB)\n")


if __name__ == "__main__":
    main()
