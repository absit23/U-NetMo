import torch
import numpy as np

class SegmentationMetrics:
    """
    Evaluates multi-class semantic segmentation metrics using a confusion matrix.
    Computes mIoU, mDice, Precision, and Recall across all classes.
    """
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Resets the internal confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def _fast_hist(self, label_true, label_pred):
        """Generates 2D confusion matrix for single batch prediction."""
        mask = (label_true >= 0) & (label_true < self.num_classes) & (label_true != self.ignore_index)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Updates confusion matrix with new batch predictions.
        
        Args:
            preds (torch.Tensor): Model logits [B, C, H, W] or predictions [B, H, W].
            targets (torch.Tensor): Ground truth masks [B, H, W].
        """
        if preds.ndim == 4:
            preds = torch.argmax(preds, dim=1)
            
        preds_np = preds.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()

        self.confusion_matrix += self._fast_hist(targets_np, preds_np)

    def compute(self):
        """
        Calculates final class-wise and mean metrics.
        
        Returns:
            dict: Summary containing mIoU, mDice, Precision, Recall, and per-class IoUs.
        """
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # Class-wise metrics
        iou = tp / (tp + fp + fn + 1e-7)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        return {
            "mIoU": float(np.nanmean(iou)),
            "mDice": float(np.nanmean(dice)),
            "mPrecision": float(np.nanmean(precision)),
            "mRecall": float(np.nanmean(recall)),
            "per_class_iou": iou.tolist(),
            "per_class_dice": dice.tolist()
        }
