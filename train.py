import os
import re
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.builder import build_model
from utils.data_loading import ConcreteDamageDataset, get_transforms
from utils.metrics import SegmentationMetrics
from utils.tracker import ExperimentTracker, CLASS_NAMES


class WeightedCombinedLoss(nn.Module):
    """Cross-entropy plus soft Dice loss for multi-class semantic segmentation."""

    def __init__(self, num_classes=8, weight_ce=1.0, weight_dice=1.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        # Expected logits: [B, C, H, W]; targets: [B, H, W] with IDs 0..C-1.
        if logits.ndim != 4:
            raise ValueError(f"Expected logits [B,C,H,W], got shape {tuple(logits.shape)}")
        if targets.ndim != 3:
            raise ValueError(f"Expected masks [B,H,W], got shape {tuple(targets.shape)}")
        if logits.shape[0] != targets.shape[0] or logits.shape[2:] != targets.shape[1:]:
            raise ValueError(
                f"Logit/mask dimensions do not match: logits={tuple(logits.shape)}, "
                f"masks={tuple(targets.shape)}. Check model output size and mask resizing."
            )
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Model outputs {logits.shape[1]} classes, but loss expects {self.num_classes}."
            )

        targets = targets.long()
        if targets.numel() and (targets.min() < 0 or targets.max() >= self.num_classes):
            raise ValueError(
                f"Mask IDs must be in [0, {self.num_classes - 1}], "
                f"but observed min={targets.min().item()}, max={targets.max().item()}."
            )

        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).to(dtype=probs.dtype)

        # Sum over batch and spatial dimensions; calculate one Dice score per class.
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        dice_per_class = (2.0 * intersection + 1e-7) / (cardinality + 1e-7)
        dice_loss = 1.0 - dice_per_class.mean()

        return self.weight_ce * ce_loss + self.weight_dice * dice_loss


def get_next_exp_name(runs_dir):
    """Return the next roundN name without overwriting previous experiment folders."""
    os.makedirs(runs_dir, exist_ok=True)
    highest_round = 0
    for name in os.listdir(runs_dir):
        match = re.fullmatch(r"round(\d+)", name)
        if match and os.path.isdir(os.path.join(runs_dir, name)):
            highest_round = max(highest_round, int(match.group(1)))
    return f"round{highest_round + 1}"


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    if len(dataloader) == 0:
        raise ValueError("Training DataLoader is empty. Check --data_dir and the training image/mask folders.")

    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, metrics_calculator, device):
    if len(dataloader) == 0:
        raise ValueError("Validation DataLoader is empty. Check the valid/images and valid/masks folders.")

    model.eval()
    running_loss = 0.0
    metrics_calculator.reset()

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        outputs = model(images)
        running_loss += criterion(outputs, masks).item()
        metrics_calculator.update(outputs, masks)

    # The metrics implementation should exclude background from the macro averages,
    # while keeping per-class arrays indexed by the original class IDs (0..C-1).
    metrics = metrics_calculator.compute(include_background=False)
    metrics["loss"] = running_loss / len(dataloader)
    return metrics


def main(args):
    if args.num_classes != 8 and args.class_weights is None:
        raise ValueError(
            "This script's default class-weight list contains 8 values. "
            "For a different --num_classes, provide matching weights or update the list in main()."
        )

    # If resuming from an explicit checkpoint without an experiment name, continue saving
    # in that checkpoint's experiment folder. Otherwise create a fresh roundN folder.
    if not args.exp_name:
        if args.resume:
            args.exp_name = os.path.basename(os.path.dirname(os.path.abspath(args.resume)))
        else:
            args.exp_name = get_next_exp_name(args.runs_dir)

    exp_dir = os.path.join(args.runs_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print(f"Experiment Output Directory: {exp_dir}\n")

    tracker = ExperimentTracker(exp_dir, args)

    # DataLoaders
    train_loader = DataLoader(
        ConcreteDamageDataset(
            os.path.join(args.data_dir, "train", "images"),
            os.path.join(args.data_dir, "train", "masks"),
            transform=get_transforms(img_size=args.img_size, is_train=True),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ConcreteDamageDataset(
            os.path.join(args.data_dir, "valid", "images"),
            os.path.join(args.data_dir, "valid", "masks"),
            transform=get_transforms(img_size=args.img_size, is_train=False),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = build_model(
        args.model_name,
        n_channels=3,
        n_classes=args.num_classes,
        pretrained=args.pretrained,
    ).to(device)

    print("=" * 70)
    print(f" MODEL ARCHITECTURE & SUMMARY: {args.model_name.upper()}")
    print("=" * 70)
    try:
        from torchinfo import summary
        summary(
            model,
            input_size=(args.batch_size, 3, args.img_size, args.img_size),
            device=device.type,
        )
    except ImportError:
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
    print("=" * 70 + "\n")

    # Loss and optimization. These weights must be checked against training-set class
    # frequencies and the exact class-ID mapping before reporting final research results.
    if args.class_weights is None:
        class_weights = [0.2, 1.0, 2.5, 6.0, 12.0, 2.0, 0.5, 1.0]
    else:
        class_weights = [float(value) for value in args.class_weights.split(",")]

    if len(class_weights) != args.num_classes:
        raise ValueError(
            f"Received {len(class_weights)} class weights for {args.num_classes} classes. "
            "Supply one weight per class, in mask-ID order."
        )

    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = WeightedCombinedLoss(
        num_classes=args.num_classes,
        weight_ce=args.weight_ce,
        weight_dice=args.weight_dice,
        class_weights=class_weights,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metrics_calculator = SegmentationMetrics(num_classes=args.num_classes)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mIoU": [],
        "val_mDice": [],
        "val_mPrecision": [],
        "val_mRecall": [],
        "per_class_iou": [],
    }
    best_miou, best_epoch, best_metrics, start_epoch = 0.0, 0, {}, 1

    # Resume checkpoint if available. An explicit --resume path takes precedence.
    resume_path = args.resume or os.path.join(exp_dir, f"last_{args.model_name}.pth")
    if os.path.isfile(resume_path):
        print(f"=> Resuming training from: '{resume_path}'")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint.get("best_miou", 0.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_metrics = checkpoint.get("best_metrics", {})
        history = checkpoint.get("history", history)

        # Backward compatibility: if resuming an older last-checkpoint, recover best
        # metrics from the separate best checkpoint when those fields are available.
        best_path = os.path.join(exp_dir, f"best_{args.model_name}.pth")
        if not best_metrics and os.path.isfile(best_path):
            best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            best_epoch = int(best_checkpoint.get("best_epoch", best_checkpoint.get("epoch", 0)))
            best_metrics = best_checkpoint.get("best_metrics", {})
            best_miou = max(best_miou, float(best_checkpoint.get("best_miou", 0.0)))

        print(f"=> Resumed from epoch {start_epoch}; best mIoU so far: {best_miou:.4f}\n")

    if start_epoch > args.epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch - 1}, which meets/exceeds "
            f"--epochs {args.epochs}. Increase --epochs to continue training."
        )

    print(f"--- Starting Training: {args.model_name.upper()} (Epochs {start_epoch}-{args.epochs}) ---")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, metrics_calculator, device)
        scheduler.step()

        tracker.log_epoch(epoch, train_loss, val_metrics, history)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"mIoU: {val_metrics['mIoU']:.4f} | "
            f"F1 (mDice): {val_metrics['mDice']:.4f} | "
            f"Precision: {val_metrics['mPrecision']:.4f} | "
            f"Recall: {val_metrics['mRecall']:.4f}"
        )

        # Update best results BEFORE saving last checkpoint so the latest best score
        # is not lost from the resume checkpoint.
        is_best = float(val_metrics["mIoU"]) > best_miou
        if is_best:
            best_miou = float(val_metrics["mIoU"])
            best_epoch = epoch
            best_metrics = val_metrics

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": best_miou,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "history": history,
        }

        last_path = os.path.join(exp_dir, f"last_{args.model_name}.pth")
        torch.save(checkpoint_state, last_path)

        if is_best:
            best_path = os.path.join(exp_dir, f"best_{args.model_name}.pth")
            torch.save(checkpoint_state, best_path)
            print(f"Saved NEW BEST model checkpoint to {best_path} (mIoU: {best_miou:.4f})")

    # Final summary. Per-class array indexing depends on utils.metrics.SegmentationMetrics:
    # its per_class_* arrays should retain original class-ID indexing even when background
    # is excluded from macro averages. Verify that implementation before relying on table.
    print("\n" + "=" * 70)
    print(f" TRAINING SUMMARY: {args.model_name.upper()} (Run: {args.exp_name})")
    print("=" * 70)
    print(f"Run Directory:        {exp_dir}")
    print(f"Best Epoch:           {best_epoch} / {args.epochs}")
    print(f"Best Val mIoU:        {best_miou:.4f} ({best_miou * 100:.2f}%)")

    if best_metrics:
        for key, label in (
            ("mDice", "Best Val F1 / mDice"),
            ("mPrecision", "Best Val Precision"),
            ("mRecall", "Best Val Recall"),
        ):
            if key in best_metrics:
                print(f"{label:<22}: {best_metrics[key]:.4f} ({best_metrics[key] * 100:.2f}%)")

        print("\n--- BEST EPOCH PER-CLASS BREAKDOWN ---")
        print(f"{'ID':<4} | {'Class Name':<24} | {'IoU':<10} | {'Dice (F1)':<10}")
        print("-" * 62)
        per_class_iou = best_metrics.get("per_class_iou", [])
        per_class_dice = best_metrics.get("per_class_dice", [])
        for cls_idx in range(args.num_classes):
            name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Class {cls_idx}"
            iou_val = per_class_iou[cls_idx] if cls_idx < len(per_class_iou) else float("nan")
            dice_val = per_class_dice[cls_idx] if cls_idx < len(per_class_dice) else float("nan")
            print(f"{cls_idx:<4} | {name:<24} | {iou_val:<10.4f} | {dice_val:<10.4f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training multi-class concrete-damage segmentation models")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root containing train/ and valid/ folders")
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet_mobilenet",
        choices=["unet", "unet_mobilenet", "deeplabv3plus", "segformer"],
    )
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_ce", type=float, default=1.0)
    parser.add_argument("--weight_dice", type=float, default=1.0)
    parser.add_argument("--class_weights", type=str, default=None, help="Optional comma-separated class weights in class-ID order")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="/content/drive/MyDrive/Colab Notebooks/U-Net/runs/segment",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment folder name. If omitted, creates the next roundN folder.",
    )

    main(parser.parse_args())
