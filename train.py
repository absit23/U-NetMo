import os
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
    def __init__(self, num_classes=8, weight_ce=1.0, weight_dice=1.0, class_weights=None):
        super().__init__()
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
        images, masks = batch['image'].to(device), batch['mask'].to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), masks)
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
        images, masks = batch['image'].to(device), batch['mask'].to(device)
        outputs = model(images)
        running_loss += criterion(outputs, masks).item()
        metrics_calculator.update(outputs, masks)

    metrics = metrics_calculator.compute(include_background=False)
    metrics["loss"] = running_loss / len(dataloader)
    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Set up unified experiment output directory: runs_dir / exp_name
    exp_dir = os.path.join(args.runs_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Initialize tracker (saves args.json and prepares metrics CSV)
    tracker = ExperimentTracker(exp_dir, args)

    print(f"Experiment Output Directory: {exp_dir}\n")

    train_loader = DataLoader(
        ConcreteDamageDataset(
            os.path.join(args.data_dir, "train", "images"),
            os.path.join(args.data_dir, "train", "masks"),
            transform=get_transforms(img_size=args.img_size, is_train=True)
        ), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        ConcreteDamageDataset(
            os.path.join(args.data_dir, "valid", "images"),
            os.path.join(args.data_dir, "valid", "masks"),
            transform=get_transforms(img_size=args.img_size, is_train=False)
        ), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = build_model(args.model_name, n_channels=3, n_classes=args.num_classes, pretrained=args.pretrained).to(device)
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
    best_miou, start_epoch = 0.0, 1

    # Resume checkpoint lookup inside exp_dir if specified or default
    resume_path = args.resume or os.path.join(exp_dir, f"last_{args.model_name}.pth")
    if os.path.isfile(resume_path):
        print(f"=> Resume checkpoint found: '{resume_path}'")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", 0.0)
        history = checkpoint.get("history", history)
        print(f"=> Resumed from epoch {start_epoch} (Best mIoU so far: {best_miou:.4f})")

    print(f"--- Training {args.model_name.upper()} (Epochs {start_epoch}-{args.epochs}) ---")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, metrics_calculator, device)
        scheduler.step()

        tracker.log_epoch(epoch, train_loss, val_metrics, history)

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | mIoU: {val_metrics['mIoU']:.4f} | "
              f"F1: {val_metrics['mDice']:.4f}")

        checkpoint_state = {
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou, 
            'history': history
        }

        # Save weights directly inside the specific round directory
        last_path = os.path.join(exp_dir, f"last_{args.model_name}.pth")
        torch.save(checkpoint_state, last_path)

        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            checkpoint_state['best_miou'] = best_miou
            best_path = os.path.join(exp_dir, f"best_{args.model_name}.pth")
            torch.save(checkpoint_state, best_path)
            print(f" Saved new best model to {best_path} (mIoU: {best_miou:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking Concrete Damage Segmentation Models")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="unet_mobilenet", choices=["unet", "unet_mobilenet", "deeplabv3plus", "segformer"])
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--runs_dir", type=str, default="/content/drive/MyDrive/Colab Notebooks/concrete_damage_checkpoints/runs/segment")
    parser.add_argument("--exp_name", type=str, default="round1")

    args = parser.parse_args()
    main(args)
