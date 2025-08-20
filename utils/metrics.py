import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Class for calculating various evaluation metrics
    Supports multi-class tasks (default 6 classes)
    """
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.reset()
        # Update default class names to fit 6 classes
        if num_classes == 6:
            self.class_names = [
                'other', 'corn', 'cotton', 'soybeans', 'spring wheat', 'winter wheat'
            ]
        elif num_classes == 4:
            self.class_names = [
                'other', 'rice', 'corn', 'soybeans'
            ]
        else:
            # For other class counts, generate generic names
            self.class_names = [f'class_{i}' for i in range(num_classes)]
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes))
        self.total_pixels = 0
        self.correct_pixels = 0
    
    def update(self, pred, target):
        """Update metrics"""
        try:
            # Ensure inputs are tensors
            if not isinstance(pred, torch.Tensor):
                pred = torch.tensor(pred)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            
            # Move to CPU
            pred = pred.cpu()
            target = target.cpu()
            
            # Check shape match
            if pred.shape != target.shape:
                raise ValueError(f"Shape mismatch: predictions {pred.shape}, targets {target.shape}")
            
            # Check value range
            if pred.min() < 0 or pred.max() >= self.num_classes:
                raise ValueError(f"Prediction values out of range [0, {self.num_classes-1}]")
            if target.min() < 0 or target.max() >= self.num_classes:
                raise ValueError(f"Target values out of range [0, {self.num_classes-1}]")
            
            # Update confusion matrix
            for t, p in zip(target.view(-1), pred.view(-1)):
                self.confusion_matrix[t.long(), p.long()] += 1
            
            # Update overall accuracy statistics
            self.total_pixels += target.numel()
            self.correct_pixels += (pred == target).sum().item()
            
        except Exception as e:
            logger.error(f"Error in metrics update: {str(e)}")
            raise
    
    def get_metrics(self):
        """Calculate all metrics"""
        try:
            metrics = {}
            
            # Calculate overall accuracy (Average Accuracy)
            metrics['accuracy'] = self.correct_pixels / self.total_pixels if self.total_pixels > 0 else 0
            
            # Safe division function
            def safe_divide(x, y, default=0.0):
                return x / y if y > 0 else default
            
            # Initialize lists for each class's metrics
            class_iou = []
            class_acc = []
            class_precision = []
            class_recall = []
            class_f1 = []
            
            # Calculate metrics for each class
            for i in range(self.num_classes):
                tp = self.confusion_matrix[i, i]
                fp = self.confusion_matrix[:, i].sum() - tp
                fn = self.confusion_matrix[i, :].sum() - tp
                tn = self.confusion_matrix.sum() - tp - fp - fn
                
                # Calculate accuracy for each class
                accuracy = safe_divide(tp + tn, self.confusion_matrix.sum())
                class_acc.append(accuracy)
                
                # Calculate precision, recall, F1 score
                precision = safe_divide(tp, tp + fp)
                recall = safe_divide(tp, tp + fn)
                f1 = safe_divide(2 * precision * recall, precision + recall)
                
                class_precision.append(precision)
                class_recall.append(recall)
                class_f1.append(f1)
                
                # Calculate IoU
                iou = safe_divide(tp, tp + fp + fn)
                class_iou.append(iou)
            
            # Calculate mean metrics
            metrics['miou'] = sum(class_iou) / len(class_iou)  # Mean IoU
            metrics['macc'] = sum(class_acc) / len(class_acc)  # Mean Accuracy
            
            # Calculate Kappa coefficient
            po = metrics['accuracy']  # Observed agreement
            pe = 0  # Expected agreement
            total = self.confusion_matrix.sum()
            
            for i in range(self.num_classes):
                row_sum = self.confusion_matrix[i, :].sum()
                col_sum = self.confusion_matrix[:, i].sum()
                pe += (row_sum * col_sum) / (total * total)
            
            metrics['kappa'] = safe_divide(po - pe, 1 - pe)
            
            # Calculate F1 Macro (average of F1 for each class)
            metrics['f1_macro'] = sum(class_f1) / len(class_f1)
            
            # Calculate F1 Weighted (weighted average F1)
            class_weights = [self.confusion_matrix[i, :].sum() / total for i in range(self.num_classes)]
            metrics['f1_weighted'] = sum(f1 * w for f1, w in zip(class_f1, class_weights))
            
            # Store metrics for each class
            metrics['iou'] = class_iou
            metrics['acc'] = class_acc
            metrics['precision'] = class_precision
            metrics['recall'] = class_recall
            metrics['f1'] = class_f1
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in metrics calculation: {str(e)}")
            # Return default metrics
            return {
                'accuracy': 0.0,
                'miou': 0.0,
                'macc': 0.0,
                'kappa': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'iou': [0.0] * self.num_classes,
                'acc': [0.0] * self.num_classes,
                'precision': [0.0] * self.num_classes,
                'recall': [0.0] * self.num_classes,
                'f1': [0.0] * self.num_classes
            } 
