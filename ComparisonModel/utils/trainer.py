import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import (
    AverageMeter, 
    accuracy, 
    compute_metrics, 
    print_metrics, 
    MetricsCalculator,
    save_metrics_to_file
)
from .checkpoint import save_checkpoint, load_checkpoint
from .data_loader import get_dataloaders


class Trainer:
    def __init__(self, model, config, model_name='model', device='cuda', save_name=None, save_last_only=False):
        self.model = model.to(device)
        self.config = config
        self.model_name = model_name
        self.device = device
        self.save_name = save_name
        self.save_last_only = save_last_only
        
        print("\nCreating dataloaders...")
        dataset_config_path = config.get('dataset_config', 'config/dataset_config.yaml')
        batch_size = config.get('batch_size', 24)
        file_list_override = config.get('file_list_override', None)
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            dataset_config_path,
            batch_size=batch_size,
            num_workers=config.get('num_workers', 4),
            file_list_override=file_list_override
        )
        
        model_class_name = self.model.__class__.__name__
        if 'Bayesian' in model_class_name or 'BBB' in model_class_name:
            from BayesianMLP import ELBOLoss
            beta = config.get('beta', 1e-6)
            
            class_weights_list = config.get('class_weights', None)
            if class_weights_list is not None:
                class_weights = torch.FloatTensor(class_weights_list).to(device)
                print(f"Bayesian model using class weights: {class_weights.cpu().numpy()}")
            else:
                class_weights = None
                print("Bayesian model: No class weights (balanced loss)")
            
            self.criterion = ELBOLoss(beta=beta, weight=class_weights)
            self.is_bayesian = True
            print(f"Using ELBO loss with beta={beta}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.is_bayesian = False
        
        lr = config.get('lr', 0.0002)
        weight_decay = config.get('weight_decay', 0.01)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.epochs = config.get('epochs', 30)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )
        
        self.start_epoch = 0
        self.best_acc = 0.0
        self.best_miou = 0.0
        
        self.checkpoint_dir = Path(config.get('checkpoint_dir', f'checkpoints/{model_name}'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.get('log_dir', f'logs/{model_name}'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.class_names = config.get('class_names', None)
        
        with open(self.checkpoint_dir / 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        print(f"\n{'='*80}")
        print(f"Trainer initialized for {model_name}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"Logs: {self.log_dir}")
        print(f"{'='*80}\n")
    
    def train_epoch(self, epoch):
        self.model.train()
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        print(f"\nEpoch [{epoch+1}/{self.epochs}]")
        print("-" * 80)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            if len(batch) == 4:
                images, labels, _, _ = batch
            else:
                images, labels = batch
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            
            is_pixel_level = len(outputs.shape) == 4
            
            if is_pixel_level:
                if self.is_bayesian:
                    kl = self.model.get_kl()
                    num_batches = len(self.train_loader)
                    loss, ce_loss, kl_loss = self.criterion(outputs, labels, kl, num_batches)
                else:
                    loss = self.criterion(outputs, labels)
                
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == labels).float().mean().item() * 100
            else:
                B = labels.shape[0]
                if len(labels.shape) == 3:
                    labels_flat = labels.reshape(B, -1)
                    labels_flat = torch.mode(labels_flat, dim=1)[0]
                else:
                    labels_flat = labels
                
                if self.is_bayesian:
                    kl = self.model.get_kl()
                    num_batches = len(self.train_loader)
                    loss, ce_loss, kl_loss = self.criterion(outputs, labels_flat, kl, num_batches)
                else:
                    loss = self.criterion(outputs, labels_flat)
                
                acc = accuracy(outputs, labels_flat)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            losses.update(loss.item(), images.size(0))
            accs.update(acc, images.size(0))
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {losses.avg:.4f} | Acc: {accs.avg:.2f}% | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        epoch_time = time.time() - start_time
        
        print(f"\nTrain Summary - Loss: {losses.avg:.4f} | Acc: {accs.avg:.2f}% | "
              f"Time: {epoch_time:.2f}s")
        
        self.writer.add_scalar('Train/Loss', losses.avg, epoch)
        self.writer.add_scalar('Train/Accuracy', accs.avg, epoch)
        self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
        
        return losses.avg, accs.avg
    
    def validate(self, epoch):
        self.model.eval()
        
        losses = AverageMeter()
        all_preds = []
        all_labels = []
        
        print("\nValidating...")
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if len(batch) == 4:
                    images, labels, _, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                
                is_pixel_level = len(outputs.shape) == 4
                
                if is_pixel_level:
                    if self.is_bayesian:
                        kl = self.model.get_kl()
                        num_batches = len(self.val_loader)
                        loss, _, _ = self.criterion(outputs, labels, kl, num_batches)
                    else:
                        loss = self.criterion(outputs, labels)
                    losses.update(loss.item(), images.size(0))
                    
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.append(preds.cpu().numpy().reshape(-1))
                    all_labels.append(labels.cpu().numpy().reshape(-1))
                else:
                    B = labels.shape[0]
                    if len(labels.shape) == 3:
                        labels_flat = labels.reshape(B, -1)
                        labels_flat = torch.mode(labels_flat, dim=1)[0]
                    else:
                        labels_flat = labels
                    
                    if self.is_bayesian:
                        kl = self.model.get_kl()
                        num_batches = len(self.val_loader)
                        loss, _, _ = self.criterion(outputs, labels_flat, kl, num_batches)
                    else:
                        loss = self.criterion(outputs, labels_flat)
                    losses.update(loss.item(), images.size(0))
                    
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels_flat.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        num_classes = self.config.get('num_classes', 6)
        metrics = compute_metrics(all_preds, all_labels, num_classes)
        
        print(f"\nValidation Summary:")
        print(f"  Loss: {losses.avg:.4f}")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  mIoU: {metrics['miou']*100:.2f}%")
        print(f"  mACC: {metrics['macc']*100:.2f}%")
        
        self.writer.add_scalar('Val/Loss', losses.avg, epoch)
        
        scalar_metrics = ['accuracy', 'miou', 'macc', 'kappa', 'f1_macro', 'f1_weighted']
        for metric_name in scalar_metrics:
            if metric_name in metrics:
                self.writer.add_scalar(f'Val/{metric_name.capitalize()}', 
                                      metrics[metric_name]*100, epoch)
        
        if 'iou_per_class' in metrics and self.class_names:
            for i, class_name in enumerate(self.class_names):
                if i < len(metrics['iou_per_class']):
                    self.writer.add_scalar(f'Val_PerClass/IoU_{class_name}', 
                                          metrics['iou_per_class'][i]*100, epoch)
        
        if 'acc_per_class' in metrics and self.class_names:
            for i, class_name in enumerate(self.class_names):
                if i < len(metrics['acc_per_class']):
                    self.writer.add_scalar(f'Val_PerClass/Acc_{class_name}', 
                                          metrics['acc_per_class'][i]*100, epoch)
        
        if 'precision_per_class' in metrics and self.class_names:
            for i, class_name in enumerate(self.class_names):
                if i < len(metrics['precision_per_class']):
                    self.writer.add_scalar(f'Val_PerClass/Precision_{class_name}', 
                                          metrics['precision_per_class'][i]*100, epoch)
        
        if 'recall_per_class' in metrics and self.class_names:
            for i, class_name in enumerate(self.class_names):
                if i < len(metrics['recall_per_class']):
                    self.writer.add_scalar(f'Val_PerClass/Recall_{class_name}', 
                                          metrics['recall_per_class'][i]*100, epoch)
        
        if 'f1_per_class' in metrics and self.class_names:
            for i, class_name in enumerate(self.class_names):
                if i < len(metrics['f1_per_class']):
                    self.writer.add_scalar(f'Val_PerClass/F1_{class_name}', 
                                          metrics['f1_per_class'][i]*100, epoch)
        
        return metrics
    
    def train(self):
        print("\n" + "="*80)
        print(f"Starting training for {self.model_name}...")
        print("="*80)
        
        for epoch in range(self.start_epoch, self.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            
            metrics = self.validate(epoch)
            
            self.scheduler.step()
            
            is_best = metrics['accuracy'] > self.best_acc
            if is_best:
                self.best_acc = metrics['accuracy']
                self.best_miou = metrics['miou']
            
            if self.save_last_only:
                if self.save_name:
                    checkpoint_filename = f'model_last_{self.save_name}.pth'
                else:
                    checkpoint_filename = 'model_last.pth'
            else:
                checkpoint_filename = 'checkpoint.pth'
                if self.save_name:
                    best_filename = f'model_best_{self.save_name}.pth'
                else:
                    best_filename = 'model_best.pth'
            
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model_name': self.model_name,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'best_acc': self.best_acc,
                    'best_miou': self.best_miou,
                    'config': self.config
                },
                is_best,
                self.checkpoint_dir,
                filename=checkpoint_filename,
                best_filename=best_filename if not self.save_last_only else None,
                save_last_only=self.save_last_only
            )
            
            print(f"\nBest Accuracy: {self.best_acc*100:.2f}% | Best mIoU: {self.best_miou*100:.2f}%")
            print("="*80)
        
        print("\nTraining completed!")
        print(f"Best Accuracy: {self.best_acc*100:.2f}%")
        print(f"Best mIoU: {self.best_miou*100:.2f}%")
        
        self.writer.close()
    
    def test(self):
        import rasterio
        from rasterio.transform import from_bounds
        
        self.model.eval()
        
        num_classes = self.config.get('num_classes', 6)
        metrics_calc = MetricsCalculator(num_classes=num_classes, class_names=self.class_names)
        
        results_dir = Path(self.config.get('results_dir', f'results/{self.model_name}'))
        predictions_dir = results_dir / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print(f"Testing {self.model_name}...")
        print("="*80 + "\n")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                if len(batch) == 4:
                    images, labels, img_paths, label_paths = batch
                else:
                    images, labels = batch
                    img_paths = [f"sample_{i}" for i in range(len(images))]
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                
                is_pixel_level = len(outputs.shape) == 4
                
                B = images.shape[0]
                
                if is_pixel_level:
                    preds = torch.argmax(outputs, dim=1)
                    
                    preds_flat = preds.cpu().numpy().reshape(-1)
                    labels_flat = labels.cpu().numpy().reshape(-1)
                    metrics_calc.update(torch.from_numpy(preds_flat), torch.from_numpy(labels_flat))
                else:
                    preds = torch.argmax(outputs, dim=1)
                    
                    if len(labels.shape) == 3:
                        labels_flat = labels.reshape(B, -1)
                        labels_flat = torch.mode(labels_flat, dim=1)[0]
                    else:
                        labels_flat = labels
                    
                    metrics_calc.update(preds.cpu(), labels_flat.cpu())
                
                for i in range(B):
                    img_path = img_paths[i]
                    
                    if is_pixel_level:
                        pred = preds[i].cpu().numpy()
                    else:
                        pred = preds[i].cpu().numpy()
                    
                    label = labels[i].cpu().numpy()
                    
                    img_filename = Path(img_path).stem
                    output_filename = f"{img_filename}_pred.tif"
                    output_path = predictions_dir / output_filename
                    
                    if len(pred.shape) == 0:
                        if len(label.shape) == 2:
                            pred_map = np.full_like(label, pred, dtype=np.uint8)
                        else:
                            pred_map = np.full((128, 128), pred, dtype=np.uint8)
                    else:
                        pred_map = pred.astype(np.uint8)
                    
                    try:
                        with rasterio.open(img_path) as src:
                            meta = src.meta.copy()
                            meta.update({
                                'count': 1,
                                'dtype': 'uint8',
                                'compress': 'lzw'
                            })
                            
                            with rasterio.open(output_path, 'w', **meta) as dst:
                                dst.write(pred_map, 1)
                    except Exception as e:
                        meta = {
                            'driver': 'GTiff',
                            'height': pred_map.shape[0],
                            'width': pred_map.shape[1],
                            'count': 1,
                            'dtype': 'uint8',
                            'compress': 'lzw'
                        }
                        with rasterio.open(output_path, 'w', **meta) as dst:
                            dst.write(pred_map, 1)
        
        metrics = metrics_calc.get_metrics()
        
        print_metrics(metrics, self.class_names)
        
        save_metrics_to_file(metrics, results_dir / 'metrics.txt', self.class_names)
        
        print(f"\n{'='*80}")
        print(f"Results saved to {results_dir}")
        print(f"  - Metrics: {results_dir / 'metrics.txt'}")
        print(f"  - Predictions: {predictions_dir}/ ({len(list(predictions_dir.glob('*.tif')))} TIF files)")
        print(f"{'='*80}\n")
        
        print(f"Testing completed!")
        print(f"Final Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Final mIoU: {metrics['miou']*100:.2f}%")
        
        return metrics
