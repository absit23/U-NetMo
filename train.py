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

    # 1. Automatic Experiment Round Folder Assignment (round1, round2, round3...)
    if not args.exp_name:
        args.exp_name = get_next_exp_name(args.runs_dir)

    exp_dir = os.path.join(args.runs_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    tracker = ExperimentTracker(exp_dir, args)
    print(f"Experiment Output Directory: {exp_dir}\n")

    # 2. DataLoaders
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

    # 3. Build Model & Print Full Architecture + Parameter Summary
    model = build_model(args.model_name, n_channels=3, n_classes=args.num_classes, pretrained=args.pretrained).to(device)

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

    # 4. Optimization & Weighted Combined Loss
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
    best_miou, best_epoch, best_metrics, start_epoch = 0.0, 0, {}, 1

    # 5. Resume Logic (Checks if checkpoint exists inside exp_dir or explicit path)
    resume_path = args.resume or os.path.join(exp_dir, f"last_{args.model_name}.pth")
    if os.path.isfile(resume_path):
        print(f"=> Resuming training from: '{resume_path}'")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", 0.0)
        history = checkpoint.get("history", history)
        print(f"=> Resumed from epoch {start_epoch} (Best mIoU so far: {best_miou:.4f})\n")

    # 6. Training Loop
    print(f"--- Starting Training: {args.model_name.upper()} (Epochs {start_epoch}-{args.epochs}) ---")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, metrics_calculator, device)
        scheduler.step()

        # Log Metrics & Update Plots in Tracker
        tracker.log_epoch(epoch, train_loss, val_metrics, history)

        # Print all 5 primary metrics on every epoch line
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
            'history': history
        }

        # Overwritten after every epoch
        last_path = os.path.join(exp_dir, f"last_{args.model_name}.pth")
        torch.save(checkpoint_state, last_path)

        # Updated only when validation mIoU achieves a new record high
        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            best_epoch = epoch
            best_metrics = val_metrics
            checkpoint_state['best_miou'] = best_miou
            best_path = os.path.join(exp_dir, f"best_{args.model_name}.pth")
            torch.save(checkpoint_state, best_path)
            print(f" Saved NEW BEST model checkpoint to {best_path} (mIoU: {best_miou:.4f})")

    # 7. Final Experiment Summary Report
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
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--runs_dir", type=str, default="/content/drive/MyDrive/Colab Notebooks/U-Net/runs/segment")
    
    # Default is None -> Automatically picks round1, round2, round3...
    parser.add_argument("--exp_name", type=str, default=None, help="Folder name for the run. Leave empty to auto-increment (round1, round2, etc.)")

    args = parser.parse_args()
    main(args)
