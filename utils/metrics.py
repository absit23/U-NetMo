import torch
import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.num_classes) & (label_true != self.ignore_index)
        return np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if preds.ndim == 4:
            preds = torch.argmax(preds, dim=1)
        preds_np = preds.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()
        self.confusion_matrix += self._fast_hist(targets_np, preds_np)

    def compute(self, include_background=False):
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        iou = tp / (tp + fp + fn + 1e-7)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        # Exclude background class 0 from mean metrics if requested
        start_idx = 0 if include_background else 1

        return {
            "mIoU": float(np.nanmean(iou[start_idx:])),
            "mDice": float(np.nanmean(dice[start_idx:])),
            "mPrecision": float(np.nanmean(precision[start_idx:])),
            "mRecall": float(np.nanmean(recall[start_idx:])),
            "per_class_iou": iou.tolist(),
            "per_class_dice": dice.tolist()
        }
