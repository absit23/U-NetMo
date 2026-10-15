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
    def __init__(self, num_classes=8, weight_ce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()

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


@torch.no_grad()
def evaluate_test_set(model, dataloader, criterion, metrics_calculator, device):
    model.eval()
    running_loss = 0.0
    metrics_calculator.reset()

    for batch in tqdm(dataloader, desc="Evaluating Test Set"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item()

        metrics_calculator.update(outputs, masks)

    metrics = metrics_calculator.compute(include_background=False)
    metrics["loss"] = running_loss / len(dataloader)
    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Check for test folder, fallback to valid if test does not exist
    test_img_dir = os.path.join(args.data_dir, "test", "images")
    test_mask_dir = os.path.join(args.data_dir, "test", "masks")
    
    if not os.path.exists(test_img_dir):
        print("⚠️ Test folder not found, falling back to Validation set for evaluation.")
        test_img_dir = os.path.join(args.data_dir, "valid", "images")
        test_mask_dir = os.path.join(args.data_dir, "valid", "masks")

    # 1. Dataset & DataLoader
    test_dataset = ConcreteDamageDataset(
        images_dir=test_img_dir,
        masks_dir=test_mask_dir,
        transform=get_transforms(img_size=args.img_size, is_train=False)
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Build Model & Load Checkpoint
    model = build_model(
        model_name=args.model_name,
        n_channels=3,
        n_classes=args.num_classes,
        pretrained=False
    ).to(device)

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: '{args.checkpoint_path}'")

    print(f"Loading best checkpoint from: '{args.checkpoint_path}'")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = CombinedLoss(num_classes=args.num_classes)
    metrics_calculator = SegmentationMetrics(num_classes=args.num_classes)

    # 3. Perform Evaluation
    print("\n" + "=" * 60)
    print(f" FINAL EVALUATION: {args.model_name.upper()}")
    print("=" * 60)
    
    metrics = evaluate_test_set(model, test_loader, criterion, metrics_calculator, device)

    # 4. Display Results
    print("\n--- OVERALL METRICS (EXCLUDING BACKGROUND) ---")
    print(f" Test Loss:     {metrics['loss']:.4f}")
    print(f" mIoU:          {metrics['mIoU']:.4f} ({metrics['mIoU']*100:.2f}%)")
    print(f" F1 / mDice:    {metrics['mDice']:.4f} ({metrics['mDice']*100:.2f}%)")
    print(f" Precision:     {metrics['mPrecision']:.4f} ({metrics['mPrecision']*100:.2f}%)")
    print(f" Recall:        {metrics['mRecall']:.4f} ({metrics['mRecall']*100:.2f}%)")

    print("\n--- PER-CLASS BREAKDOWN ---")
    print(f"{'Class ID':<10} | {'IoU':<10} | {'Dice (F1)':<10}")
    print("-" * 38)
    
    per_class_iou = metrics.get('per_class_iou', [])
    per_class_dice = metrics.get('per_class_dice', [])

    for cls_idx in range(args.num_classes):
        label = "Background" if cls_idx == 0 else f"Damage Class {cls_idx}"
        iou_val = per_class_iou[cls_idx] if cls_idx < len(per_class_iou) else 0.0
        dice_val = per_class_dice[cls_idx] if cls_idx < len(per_class_dice) else 0.0
        print(f"{label:<10} | {iou_val:<10.4f} | {dice_val:<10.4f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Concrete Damage Segmentation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Roboflow dataset directory")
    parser.add_argument("--model_name", type=str, default="unet_mobilenet")
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to .pth checkpoint file")

    args = parser.parse_args()
    main(args)
