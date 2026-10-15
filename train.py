import os
import json
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.builder import build_model
from utils.data_loading import ConcreteDamageDataset, get_transforms
from utils.metrics import SegmentationMetrics

# Semantic class names mapped directly to IDs 0 through 7
CLASS_NAMES = [
    "Background",
    "Delamination",
    "Efflorescence",
    "Microcrack",
    "Minor spalling",
    "Moderate spalling",
    "Rebar corrosion",
    "Structural crack"
]


class WeightedCombinedLoss(nn.Module):
    """
    Combines Weighted Cross-Entropy Loss with Soft Dice Loss.
    Applies class weights to heavily penalize errors on rare damage categories.
    """
    def __init__(self, num_classes=8, weight_ce=1.0, weight_dice=1.0, class_weights=None):
        super(WeightedCombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        targets = targets.long()
        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        dice_loss = 1.0 - torch.mean((2.0 * intersection + 1e-7) / (cardinality + 1e-7))

        return self.weight_ce * ce_loss + self.weight_dice * dice_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, metrics_calculator, device):
    model.eval()
    running_loss = 0.0
    metrics_calculator.reset()

    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item()

        metrics_calculator.update(outputs, masks)

    metrics = metrics_calculator.compute(include_background=False)
    metrics["loss"] = running_loss / len(dataloader)
    return metrics


def save_args(args, exp_dir):
    """Saves training hyperparameter configuration as args.json."""
    args_dict = vars(args)
    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)


def append_metrics_csv(csv_path, row_data, is_first=False):
    """Logs epoch metrics into a CSV spreadsheet."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists or is_first:
            headers = [
                "epoch", "train_loss", "val_loss", "val_mIoU", "val_mDice", 
                "val_mPrecision", "val_mRecall"
            ] + [f"IoU_{name}" for name in CLASS_NAMES]
            writer.writerow(headers)
        writer.writerow(row_data)


def generate_and_save_plots(history, exp_dir):
    """Generates loss, mIoU, precision/recall, and per-class IoU curves."""
    epochs = history['epoch']

    # 1. Training & Validation Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='tab:blue', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='tab:orange', linewidth=2)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # 2. Overall Validation Metrics Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['val_mIoU'], label='mIoU', color='tab:green', linewidth=2)
    plt.plot(epochs, history['val_mDice'], label='mDice (F1)', color='tab:red', linewidth=2)
    plt.plot(epochs, history['val_mPrecision'], label='Precision', color='tab:purple', linewidth=2)
    plt.plot(epochs, history['val_mRecall'], label='Recall', color='tab:brown', linewidth=2)
    plt.title('Validation Performance Metrics (Excl. Background)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "metrics_curve.png"), dpi=300)
    plt.close()

    # 3. Per-Class IoU Trajectory Plot
    plt.figure(figsize=(10, 6))
    for idx, cls_name in enumerate(CLASS_NAMES):
        if cls_name == "Background":
            continue
        iou_series = [h[idx] for h in history['per_class_iou']]
        plt.plot(epochs, iou_series, label=cls_name, linewidth=1.8)
    plt.title('Per-Class IoU Trajectories Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "per_class_iou_curve.png"), dpi=300)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Experiment Output Directory Setup
    exp_dir = os.path.join(args.runs_dir, args.exp_name)
    weights_dir = os.path.join(exp_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Save hyperparameter configuration
    save_args(args, exp_dir)
    csv_path = os.path.join(exp_dir, "metrics.csv")

    print(f"Experiment artifacts will be saved to: {exp_dir}\n")

    # 1. Dataset & DataLoaders
    train_dataset = ConcreteDamageDataset(
        images_dir=os.path.join(args.data_dir, "train", "images"),
        masks_dir=os.path.join(args.data_dir, "train", "masks"),
        transform=get_transforms(img_size=args.img_size, is_train=True)
    )
    val_dataset = ConcreteDamageDataset(
        images_dir=os.path.join(args.data_dir, "valid", "images"),
        masks_dir=os.path.join(args.data_dir, "valid", "masks"),
        transform=get_transforms(img_size=args.img_size, is_train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Build Model
    model = build_model(
        model_name=args.model_name,
        n_channels=3,
        n_classes=args.num_classes,
        pretrained=args.pretrained
    ).to(device)

    # Summary
    print("=" * 70)
    print(f" MODEL ARCHITECTURE & SUMMARY: {args.model_name.upper()}")
    print("=" * 70)
    try:
        from torchinfo import summary
        summary(model, input_size=(args.batch_size, 3, args.img_size, args.img_size), device=device.type)
    except ImportError:
        print(model)
        print("-" * 70)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
    print("=" * 70 + "\n")

    # 3. Class Weights
    class_weights = torch.tensor([0.2, 1.0, 2.5, 6.0, 12.0, 2.0, 0.5, 1.0], device=device)

    criterion = WeightedCombinedLoss(num_classes=args.num_classes, class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    metrics_calculator = SegmentationMetrics(num_classes=args.num_classes)

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 
        'val_mIoU': [], 'val_mDice': [], 'val_mPrecision': [], 
        'val_mRecall': [], 'per_class_iou': []
    }

    best_miou = 0.0
    best_epoch = 0
    best_metrics = {}
    start_epoch = 1

    # 4. Resume Checkpoint Logic
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_miou = checkpoint.get("best_miou", 0.0)
            if "history" in checkpoint:
                history = checkpoint["history"]
            print(f"=> Resumed from epoch {start_epoch}; best mIoU so far: {best_miou:.4f}")

    print(f"--- Starting Training: {args.model_name} (Epochs {start_epoch} to {args.epochs}) ---")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, metrics_calculator, device)
        scheduler.step()

        # Update History
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mIoU'].append(val_metrics['mIoU'])
        history['val_mDice'].append(val_metrics['mDice'])
        history['val_mPrecision'].append(val_metrics['mPrecision'])
        history['val_mRecall'].append(val_metrics['mRecall'])
        history['per_class_iou'].append(val_metrics['per_class_iou'])

        # Save Metrics CSV
        row = [
            epoch, f"{train_loss:.4f}", f"{val_metrics['loss']:.4f}", 
            f"{val_metrics['mIoU']:.4f}", f"{val_metrics['mDice']:.4f}", 
            f"{val_metrics['mPrecision']:.4f}", f"{val_metrics['mRecall']:.4f}"
        ] + [f"{iou:.4f}" for iou in val_metrics['per_class_iou']]
        append_metrics_csv(csv_path, row, is_first=(epoch == 1))

        # Generate Updated Figures
        generate_and_save_plots(history, exp_dir)

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"mIoU: {val_metrics['mIoU']:.4f} | "
              f"F1 (mDice): {val_metrics['mDice']:.4f} | "
              f"Precision: {val_metrics['mPrecision']:.4f} | "
              f"Recall: {val_metrics['mRecall']:.4f}")

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou,
            'val_metrics': val_metrics,
            'history': history
        }

        last_path = os.path.join(weights_dir, "last.pth")
        torch.save(checkpoint_state, last_path)

        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            best_epoch = epoch
            best_metrics = val_metrics
            checkpoint_state['best_miou'] = best_miou
            best_path = os.path.join(weights_dir, "best.pth")
            torch.save(checkpoint_state, best_path)
            print(f" Saved new best model checkpoint to {best_path} (mIoU: {best_miou:.4f})")

    # Final Summary Report
    print("\n" + "=" * 70)
    print(f" TRAINING COMPLETE: {args.model_name.upper()} (Run: {args.exp_name})")
    print("=" * 70)
    print(f" Run Directory:        {exp_dir}")
    print(f" Best Epoch:           {best_epoch} / {args.epochs}")
    print(f" Best Val mIoU:        {best_miou:.4f} ({best_miou*100:.2f}%)")
    if best_metrics:
        print(f" Best Val F1 / mDice:  {best_metrics['mDice']:.4f} ({best_metrics['mDice']*100:.2f}%)")
        print(f" Best Val Precision:   {best_metrics['mPrecision']:.4f} ({best_metrics['mPrecision']*100:.2f}%)")
        print(f" Best Val Recall:      {best_metrics['mRecall']:.4f} ({best_metrics['mRecall']*100:.2f}%)")

        print("\n--- BEST EPOCH PER-CLASS BREAKDOWN ---")
        print(f"{'ID':<4} | {'Class Name':<20} | {'IoU':<10} | {'Dice (F1)':<10}")
        print("-" * 55)
        per_class_iou = best_metrics.get('per_class_iou', [])
        per_class_dice = best_metrics.get('per_class_dice', [])
        for cls_idx in range(args.num_classes):
            name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Class {cls_idx}"
            iou_val = per_class_iou[cls_idx] if cls_idx < len(per_class_iou) else 0.0
            dice_val = per_class_dice[cls_idx] if cls_idx < len(per_class_dice) else 0.0
            print(f"{cls_idx:<4} | {name:<20} | {iou_val:<10.4f} | {dice_val:<10.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking Concrete Damage Segmentation Models")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Roboflow dataset directory")
    parser.add_argument("--model_name", type=str, default="unet_mobilenet", choices=["unet", "unet_mobilenet", "deeplabv3plus", "segformer"])
    parser.add_argument("--num_classes", type=int, default=8, help="Number of segmentation classes including background")
    parser.add_argument("--img_size", type=int, default=512, help="Image spatial resolution")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use pretrained encoder weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    
    # Run directory settings
    parser.add_argument("--runs_dir", type=str, default="/content/drive/MyDrive/Colab Notebooks/concrete_damage_checkpoints/runs/segment", help="Base directory for experiment runs")
    parser.add_argument("--exp_name", type=str, default="round1", help="Unique name for this training run (e.g., round1, round2_deeplab)")

    args = parser.parse_args()
    main(args)
