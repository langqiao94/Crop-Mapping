import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsCalculator:
    def __init__(self, num_classes=6, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes))
        self.total_pixels = 0
        self.correct_pixels = 0
    
    def update(self, pred, target):
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)
        
        pred = pred.cpu()
        target = target.cpu()
        
        if pred.min() < 0 or pred.max() >= self.num_classes:
            print(f"Warning: Prediction values out of range [0, {self.num_classes-1}]")
        if target.min() < 0 or target.max() >= self.num_classes:
            print(f"Warning: Target values out of range [0, {self.num_classes-1}]")
        
        for t, p in zip(target.view(-1), pred.view(-1)):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t.long(), p.long()] += 1
        
        self.total_pixels += target.numel()
        self.correct_pixels += (pred == target).sum().item()
    
    def get_metrics(self):
        metrics = {}
        
        metrics['accuracy'] = self.correct_pixels / self.total_pixels if self.total_pixels > 0 else 0
        
        def safe_divide(x, y, default=0.0):
            return x / y if y > 0 else default
        
        class_iou = []
        class_acc = []
        class_precision = []
        class_recall = []
        class_f1 = []
        
        cm = self.confusion_matrix.numpy() if isinstance(self.confusion_matrix, torch.Tensor) else self.confusion_matrix
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            accuracy = safe_divide(tp + tn, cm.sum())
            class_acc.append(accuracy)
            
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            f1 = safe_divide(2 * precision * recall, precision + recall)
            
            class_precision.append(precision)
            class_recall.append(recall)
            class_f1.append(f1)
            
            iou = safe_divide(tp, tp + fp + fn)
            class_iou.append(iou)
        
        metrics['miou'] = sum(class_iou) / len(class_iou)
        metrics['macc'] = sum(class_acc) / len(class_acc)
        
        po = metrics['accuracy']
        pe = 0
        total = cm.sum()
        
        for i in range(self.num_classes):
            row_sum = cm[i, :].sum()
            col_sum = cm[:, i].sum()
            pe += (row_sum * col_sum) / (total * total)
        
        metrics['kappa'] = safe_divide(po - pe, 1 - pe)
        
        metrics['f1_macro'] = sum(class_f1) / len(class_f1)
        
        class_weights = [cm[i, :].sum() / total for i in range(self.num_classes)]
        metrics['f1_weighted'] = sum(f1 * w for f1, w in zip(class_f1, class_weights))
        
        metrics['iou_per_class'] = class_iou
        metrics['acc_per_class'] = class_acc
        metrics['precision_per_class'] = class_precision
        metrics['recall_per_class'] = class_recall
        metrics['f1_per_class'] = class_f1
        
        metrics['confusion_matrix'] = cm
        
        return metrics


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum().item()
        total = target.size(0)
        return correct / total * 100.0


def compute_metrics(predictions, labels, num_classes=6, class_names=None):
    calculator = MetricsCalculator(num_classes, class_names)
    calculator.update(predictions, labels)
    metrics = calculator.get_metrics()
    
    return metrics


def print_metrics(metrics, class_names=None):
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"  mIoU:        {metrics['miou']*100:.2f}%")
    print(f"  mACC:        {metrics['macc']*100:.2f}%")
    print(f"  Kappa:       {metrics['kappa']:.4f}")
    print(f"  F1 Macro:    {metrics['f1_macro']*100:.2f}%")
    print(f"  F1 Weighted: {metrics['f1_weighted']*100:.2f}%")
    
    if 'iou_per_class' in metrics:
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'IoU':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("-" * 80)
        
        iou_list = metrics['iou_per_class']
        acc_list = metrics['acc_per_class']
        prec_list = metrics['precision_per_class']
        recall_list = metrics['recall_per_class']
        f1_list = metrics['f1_per_class']
        
        for i in range(len(iou_list)):
            name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            print(f"{name:<20} {iou_list[i]*100:>7.2f}% {acc_list[i]*100:>7.2f}% "
                  f"{prec_list[i]*100:>7.2f}% {recall_list[i]*100:>7.2f}% {f1_list[i]*100:>7.2f}%")
    
    print("="*80 + "\n")


def save_metrics_to_file(metrics, output_path, class_names=None):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Accuracy:    {metrics['accuracy']*100:.2f}%\n")
        f.write(f"  mIoU:        {metrics['miou']*100:.2f}%\n")
        f.write(f"  mACC:        {metrics['macc']*100:.2f}%\n")
        f.write(f"  Kappa:       {metrics['kappa']:.4f}\n")
        f.write(f"  F1 Macro:    {metrics['f1_macro']*100:.2f}%\n")
        f.write(f"  F1 Weighted: {metrics['f1_weighted']*100:.2f}%\n\n")
        
        if 'iou_per_class' in metrics:
            f.write("Per-Class Metrics:\n")
            f.write(f"{'Class':<20} {'IoU':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}\n")
            f.write("-" * 80 + "\n")
            
            iou_list = metrics['iou_per_class']
            acc_list = metrics['acc_per_class']
            prec_list = metrics['precision_per_class']
            recall_list = metrics['recall_per_class']
            f1_list = metrics['f1_per_class']
            
            for i in range(len(iou_list)):
                name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                f.write(f"{name:<20} {iou_list[i]*100:>7.2f}% {acc_list[i]*100:>7.2f}% "
                       f"{prec_list[i]*100:>7.2f}% {recall_list[i]*100:>7.2f}% {f1_list[i]*100:>7.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
