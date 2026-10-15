import os
import json
import csv
import matplotlib
matplotlib.use('Agg')  # Force headless backend to prevent Colab plotting crashes
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "Background", "Delamination", "Efflorescence", "Microcrack",
    "Minor spalling", "Moderate spalling", "Rebar corrosion", "Structural crack"
]

class ExperimentTracker:
    def __init__(self, exp_dir, args):
        self.exp_dir = exp_dir
        self.csv_path = os.path.join(self.exp_dir, "metrics.csv")
        
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Save hyperparameter configuration
        with open(os.path.join(self.exp_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    def log_epoch(self, epoch, train_loss, val_metrics, history):
        # Convert tensor values safely to Python floats
        t_loss = float(train_loss)
        v_loss = float(val_metrics['loss'])
        v_miou = float(val_metrics['mIoU'])
        v_mdice = float(val_metrics['mDice'])
        v_mprec = float(val_metrics['mPrecision'])
        v_mrec = float(val_metrics['mRecall'])
        per_class_iou = [float(x) for x in val_metrics.get('per_class_iou', [])]

        history['epoch'].append(int(epoch))
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_mIoU'].append(v_miou)
        history['val_mDice'].append(v_mdice)
        history['val_mPrecision'].append(v_mprec)
        history['val_mRecall'].append(v_mrec)
        history['per_class_iou'].append(per_class_iou)

        # Append to CSV
        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                headers = [
                    "epoch", "train_loss", "val_loss", "val_mIoU", "val_mDice", 
                    "val_mPrecision", "val_mRecall"
                ] + [f"IoU_{name}" for name in CLASS_NAMES]
                writer.writerow(headers)
            
            row = [
                epoch, f"{t_loss:.4f}", f"{v_loss:.4f}", 
                f"{v_miou:.4f}", f"{v_mdice:.4f}", 
                f"{v_mprec:.4f}", f"{v_mrec:.4f}"
            ] + [f"{iou:.4f}" for iou in per_class_iou]
            writer.writerow(row)

        self.plot_curves(history)

    def plot_curves(self, history):
        epochs = history['epoch']

        # Loss Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history['train_loss'], label='Train Loss', color='tab:blue', linewidth=2)
        plt.plot(epochs, history['val_loss'], label='Val Loss', color='tab:orange', linewidth=2)
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "loss_curve.png"), dpi=300)
        plt.close()

        # Metrics Curve
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
        plt.savefig(os.path.join(self.exp_dir, "metrics_curve.png"), dpi=300)
        plt.close()

        # Per-Class IoU Trajectory Curve
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
        plt.savefig(os.path.join(self.exp_dir, "per_class_iou_curve.png"), dpi=300)
        plt.close()
