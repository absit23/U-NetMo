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


class CombinedLoss(nn.Module):
    """
    Combines Cross-Entropy Loss with Soft Dice Loss for robust multi-class segmentation.
    Helps mitigate class imbalance in concrete damage segmentation.
    """
    def __init__(self, num_classes=6, weight_ce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
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

    metrics = metrics_calculator.compute()
    metrics["loss"] = running_loss / len(dataloader)
    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

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

    # 3. Loss, Optimizer & Scheduler
    criterion = CombinedLoss(num_classes=args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    metrics_calculator = SegmentationMetrics(num_classes=args.num_classes)

    best_miou = 0.0
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
            print(f"=> Resumed from epoch {start_epoch}; best mIoU so far: {best_miou:.4f}")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")

    print(f"\n--- Starting Training: {args.model_name} (Epochs {start_epoch} to {args.epochs}) ---")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, metrics_calculator, device)
        scheduler.step()

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val mIoU: {val_metrics['mIoU']:.4f} | "
              f"Val mDice: {val_metrics['mDice']:.4f}")

        # State dict payload
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou,
        }

        # Save latest state for interruption recovery
        last_path = os.path.join(args.checkpoint_dir, f"last_{args.model_name}.pth")
        torch.save(checkpoint_state, last_path)

        # Save best model checkpoint
        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            checkpoint_state['best_miou'] = best_miou
            best_path = os.path.join(args.checkpoint_dir, f"best_{args.model_name}.pth")
            torch.save(checkpoint_state, best_path)
            print(f" Saved new best model checkpoint to {best_path} (mIoU: {best_miou:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking Concrete Damage Segmentation Models")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Roboflow dataset directory")
    parser.add_argument("--model_name", type=str, default="unet_mobilenet", choices=["unet", "unet_mobilenet", "deeplabv3plus", "segformer"])
    parser.add_argument("--num_classes", type=int, default=6, help="Number of segmentation classes")
    parser.add_argument("--img_size", type=int, default=512, help="Image spatial resolution")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use pretrained encoder weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save weights")

    args = parser.parse_args()
    main(args)
