import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
from .metrics import MetricsCalculator
import os
from types import SimpleNamespace
import matplotlib.pyplot as plt
import math

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, device, start_epoch=0):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.start_epoch = start_epoch
        
        self.base_lr = config.training.optimizer.lr
        self.warmup_epochs = config.training.lr_scheduler.warmup.epochs

        param_groups = []
        base_params = {'params': [], 'lr': self.base_lr, 'weight_decay': config.training.optimizer.weight_decay}
        no_decay_params = {'params': [], 'lr': self.base_lr, 'weight_decay': 0.0}
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in ['absolute_pos_embed', 'relative_position_bias_table']) or 'norm' in name:
                no_decay_params['params'].append(param)
            else:
                base_params['params'].append(param)
        
        param_groups.append(base_params)
        if no_decay_params['params']:
             param_groups.append(no_decay_params)
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.base_lr,
            betas=config.training.optimizer.betas,
            eps=config.training.optimizer.eps
        )
        
        epochs_after_warmup = config.training.epochs - self.warmup_epochs
        if epochs_after_warmup <= 0:
             self.logger.warning(f"Warmup epochs ({self.warmup_epochs}) >= total epochs ({config.training.epochs}). Cosine annealing will not run.")
             t_max_cosine = 1
        else:
             t_max_cosine = epochs_after_warmup

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=t_max_cosine,
            eta_min=1e-6
        )
        
        self.checkpoint_dir = Path(config.training.checkpoint.save_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.setup_logging()
        
        try:
            num_classes = config.dataset.data_info.num_classes
            self.class_names = config.dataset.data_info.class_names
            self.metrics_calculator = MetricsCalculator(num_classes=num_classes)
            self.logger.info(f"Initialized MetricsCalculator with {num_classes} classes: {self.class_names}")
        except AttributeError:
            self.metrics_calculator = MetricsCalculator(num_classes=6)
            self.class_names = ['other', 'corn', 'cotton', 'soybeans', 'spring wheat', 'winter wheat']
            self.logger.warning("Could not find num_classes or class_names in config. Using default 6-class values.")
        
        self.best_metric = float('-inf')
        self.patience_counter = 0
        self.early_stopping_patience = getattr(self.config.training.early_stopping, 'patience', 15)
        self.early_stopping_min_delta = getattr(self.config.training.early_stopping, 'min_delta', 0.0)
        
        self.log_config_and_params()
        
        if hasattr(self.config.training, 'curriculum') and self.config.training.curriculum:
            self.setup_curriculum_learning()
        
        self.epochs_completed = []
        self.train_loss_history = {'total': [], 'fusion': [], 'swin': [], 'gru': []}
        self.val_loss_history = {'total': [], 'fusion': [], 'swin': [], 'gru': []}
        self.val_acc_history = []
        self.val_miou_history = []
        self.lr_history = []

        self.plot_frequency = getattr(self.config.training, 'plot_freq', 1)
        if self.plot_frequency is None:
             self.plot_frequency = 0
        
        self.load_checkpoint()
        
    def setup_logging(self):
        import sys
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('training.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
            'epochs_completed': self.epochs_completed,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'val_acc_history': self.val_acc_history,
            'val_miou_history': self.val_miou_history,
            'lr_history': self.lr_history
        }
        
        save_freq = getattr(self.config.training.checkpoint, 'save_freq', 1)
        if epoch % save_freq == 0:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            try:
                 torch.save(checkpoint, path)
                 self.logger.info(f'Saved checkpoint: {path}')
            except Exception as e:
                 self.logger.error(f"Failed to save checkpoint {path}: {e}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            try:
                 torch.save(checkpoint, best_path)
                 self.logger.info(f'Saved best model: {best_path} (Metric: {self.config.metrics.main_metric}={self.best_metric:.4f})')
            except Exception as e:
                 self.logger.error(f"Failed to save best model checkpoint {best_path}: {e}")
        
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        try:
             torch.save(checkpoint, last_path)
        except Exception as e:
             self.logger.error(f"Failed to save last checkpoint {last_path}: {e}")

    def load_checkpoint(self):
        load_dir = Path("./His-Checkpoints")
        checkpoint_to_load_path = load_dir / 'last_checkpoint.pth'

        if checkpoint_to_load_path.exists():
            try:
                checkpoint_data = torch.load(checkpoint_to_load_path, map_location=self.device)
                self.logger.info(f"Found specified checkpoint for loading: {checkpoint_to_load_path}. Loading model weights ONLY for fine-tuning.")

                if 'model_state_dict' in checkpoint_data:
                    self.model.load_state_dict(checkpoint_data['model_state_dict'])
                    self.logger.info("Successfully loaded model weights from checkpoint.")
                else:
                    self.logger.warning(f"Checkpoint {checkpoint_to_load_path} is missing 'model_state_dict'. Model weights not loaded.")

                self.start_epoch = 0
                self.logger.info(f"Fine-tuning mode: Resetting start epoch to 0. Training will begin from epoch 1.")

                self.epochs_completed = checkpoint_data.get('epochs_completed', [])
                self.train_loss_history = checkpoint_data.get('train_loss_history', {'total': [], 'fusion': [], 'swin': [], 'gru': []})
                self.val_loss_history = checkpoint_data.get('val_loss_history', {'total': [], 'fusion': [], 'swin': [], 'gru': []})
                self.val_acc_history = checkpoint_data.get('val_acc_history', [])
                self.val_miou_history = checkpoint_data.get('val_miou_history', [])
                self.lr_history = checkpoint_data.get('lr_history', [])
                if self.epochs_completed:
                    self.logger.info("Note: Loaded previous training history, but fine-tuning will start new history from epoch 1.")

            except Exception as e:
                 self.logger.error(f"Error loading or applying specified checkpoint {checkpoint_to_load_path}: {e}. Starting from scratch (epoch 1).")
                 self.start_epoch = 0
                 self.epochs_completed = []
                 self.train_loss_history = {'total': [], 'fusion': [], 'swin': [], 'gru': []}
                 self.val_loss_history = {'total': [], 'fusion': [], 'swin': [], 'gru': []}
                 self.val_acc_history = []
                 self.val_miou_history = []
                 self.lr_history = []
        else:
            self.logger.info(f'Specified checkpoint {checkpoint_to_load_path} not found. Starting training from scratch (epoch 1).')
            self.start_epoch = 0
            self.epochs_completed = []
            self.train_loss_history = {'total': [], 'fusion': [], 'swin': [], 'gru': []}
            self.val_loss_history = {'total': [], 'fusion': [], 'swin': [], 'gru': []}
            self.val_acc_history = []
            self.val_miou_history = []
            self.lr_history = []

    def train_epoch(self, epoch):
        self.model.train()
        self.metrics_calculator.reset()
        accumulated_losses = {'total_loss': 0.0, 'fusion_loss': 0.0, 'swin_loss': 0.0, 'gru_loss': 0.0}
        num_batches = len(self.train_loader)
        epoch_samples = 0

        warmup_iters = self.warmup_epochs * num_batches

        pbar = tqdm(
            total=num_batches,
            desc=f'Epoch {epoch}/{self.config.training.epochs}',
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        current_lr = self.base_lr

        for batch_idx, batch in enumerate(self.train_loader):
            global_step = (epoch - 1) * num_batches + batch_idx

            if epoch <= self.warmup_epochs:
                if warmup_iters > 0:
                     lr_scale = min(1., float(global_step + 1) / float(warmup_iters))
                     current_warmup_lr = self.base_lr * lr_scale
                     for param_group in self.optimizer.param_groups:
                         param_group['lr'] = current_warmup_lr
                     current_lr = current_warmup_lr
                
                if global_step % (num_batches // 4 + 1) == 0:
                     self.logger.debug(f"Epoch [{epoch}/{self.warmup_epochs}] Step [{global_step}/{warmup_iters}] Warmup LR: {current_lr:.8f}")
            else:
                 current_lr = self.optimizer.param_groups[0]['lr']

            try:
                required_keys = ['hls', 'history', 'label']
                if not isinstance(batch, dict) or any(k not in batch for k in required_keys):
                    self.logger.error(f"Train Batch {batch_idx} invalid. Skipping.")
                    pbar.update(1)
                    continue
                
                model_input_batch = {k: batch[k].to(self.device) for k in ['hls', 'history']}
                labels = batch['label'].to(self.device)
                batch_size = labels.size(0)
                epoch_samples += batch_size

                outputs = self.model(model_input_batch)

                if not isinstance(outputs, dict) or not all(k in outputs for k in ['output', 'swin_output', 'gru_output']):
                     self.logger.error(f"Train Batch {batch_idx}: Invalid model output. Skipping.")
                     pbar.update(1)
                     continue
                loss, loss_dict = self.model.forward_loss(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()

                clip_grad_cfg = getattr(self.config.training, 'clip_grad', None)
                if clip_grad_cfg:
                    max_norm = getattr(clip_grad_cfg, 'max_norm', 1.0)
                    norm_type = getattr(clip_grad_cfg, 'norm_type', 2)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=norm_type)

                self.optimizer.step()

                for k, v_loss in loss_dict.items():
                    if k in accumulated_losses:
                        accumulated_losses[k] += v_loss * batch_size
                    else:
                        self.logger.warning(f"Loss key '{k}' not tracked.")
                
                logits = outputs.get('output')
                if logits is not None:
                    predictions_idx = logits.argmax(dim=1)
                    self.metrics_calculator.update(predictions_idx.detach(), labels.detach())
                else:
                    self.logger.error("Missing 'output' in model outputs for metrics.")

                pbar.update(1)
                avg_total_loss = accumulated_losses['total_loss'] / epoch_samples if epoch_samples > 0 else 0
                pbar.set_postfix({
                    'loss': f'{avg_total_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })

            except Exception as e:
                self.logger.error(f"Error during train batch {batch_idx}: {type(e).__name__} - {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                pbar.update(1)
                continue

        pbar.close()
        
        self.lr_history.append(current_lr)

        metrics = self.metrics_calculator.get_metrics()
        if epoch_samples > 0:
             avg_losses = {k: v / epoch_samples for k, v in accumulated_losses.items()}
             metrics.update(avg_losses)
        else:
             self.logger.warning("No samples processed during training epoch.")
             metrics.update({k: 0.0 for k in accumulated_losses})

        return metrics
    
    def validate(self):
        self.model.eval()
        self.metrics_calculator.reset()
        total_loss = 0
        accumulated_losses = {'total_loss': 0.0, 'fusion_loss': 0.0, 'swin_loss': 0.0, 'gru_loss': 0.0}
        num_batches = len(self.val_loader)
        epoch_samples = 0

        with torch.no_grad():
            pbar = tqdm(total=num_batches, desc='Validation', leave=False)
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    required_keys = ['hls', 'history', 'label']
                    if not isinstance(batch, dict):
                        self.logger.error(f"Validation Batch {batch_idx} is not a dictionary. Skipping.")
                        pbar.update(1)
                        continue
                    missing_keys = [key for key in required_keys if key not in batch]
                    if missing_keys:
                        self.logger.error(f"Validation Batch {batch_idx} missing required keys: {missing_keys}. Available: {list(batch.keys())}. Skipping.")
                        pbar.update(1)
                        continue

                    model_input_batch = {}
                    for key in ['hls', 'history']:
                        model_input_batch[key] = batch[key].to(self.device)
                    labels = batch['label'].to(self.device)
                    batch_size = labels.size(0)
                    epoch_samples += batch_size

                    outputs = self.model(model_input_batch)

                    if not isinstance(outputs, dict):
                        self.logger.error(f"Validation error in batch {batch_idx}: Model output is not a dictionary (type: {type(outputs)}). Skipping.")
                        pbar.update(1)
                        continue

                    logits = outputs.get('output')
                    if logits is None:
                        logits = outputs.get('predictions')

                    if logits is None:
                        self.logger.error(f"Validation error in batch {batch_idx}: Model output dictionary missing 'output' or 'predictions' key (logits). Keys found: {list(outputs.keys())}. Skipping.")
                        pbar.update(1)
                        continue

                    try:
                        required_loss_keys = ['output', 'swin_output', 'gru_output']
                        if all(k in outputs for k in required_loss_keys):
                             loss, loss_dict = self.model.forward_loss(outputs, labels)
                             total_loss += loss.item() * batch_size
                             for k, v in loss_dict.items():
                                if k in accumulated_losses:
                                     accumulated_losses[k] += v * batch_size
                                else:
                                     self.logger.warning(f"Loss key '{k}' not tracked in validation.")
                        else:
                             missing_loss_keys = [k for k in required_loss_keys if k not in outputs]
                             self.logger.warning(f"Could not calculate validation loss for batch {batch_idx}: Missing keys in model output: {missing_loss_keys}")
                             loss_dict = {k: float('nan') for k in accumulated_losses}

                    except Exception as loss_e:
                         self.logger.warning(f"Error calculating validation loss for batch {batch_idx}: {type(loss_e).__name__} - {loss_e}")
                         loss_dict = {k: float('nan') for k in accumulated_losses}

                    predictions_idx = logits.argmax(dim=1)
                    self.metrics_calculator.update(predictions_idx.detach(), labels.detach())

                    pbar.update(1)
                    if epoch_samples > 0:
                         avg_val_total_loss = accumulated_losses['total_loss'] / epoch_samples if not math.isnan(accumulated_losses['total_loss']) else float('nan')
                         pbar.set_postfix({'val_loss': f'{avg_val_total_loss:.4f}'})

                except Exception as e:
                    self.logger.error(f"Unexpected error during validation batch {batch_idx}: {type(e).__name__} - {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    pbar.update(1)
                    continue

            pbar.close()

        metrics = self.metrics_calculator.get_metrics()
        if epoch_samples > 0:
            avg_losses = {k: (v / epoch_samples) if not math.isnan(v) else float('nan') for k, v in accumulated_losses.items()}
            metrics.update(avg_losses)
        else:
             self.logger.warning("No samples processed during validation.")
             metrics.update({k: 0.0 for k in accumulated_losses})

        self.model.train()

        return metrics

    def log_config_and_params(self):
        config_log_path = os.path.join(self.config.training.checkpoint.save_dir, 'config_log.txt')
        os.makedirs(os.path.dirname(config_log_path), exist_ok=True)
        
        with open(config_log_path, 'w', encoding='utf-8') as f:
            self.logger.info("=" * 20 + " Configuration Information " + "=" * 20)
            f.write("=" * 20 + " Configuration Information " + "=" * 20 + "\n")
            
            def log_namespace(ns, prefix="", file_handle=None, logger_handle=None):
                items = vars(ns)
                for key, value in items.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, SimpleNamespace):
                        log_namespace(value, prefix=full_key, file_handle=file_handle, logger_handle=logger_handle)
                    elif isinstance(value, (str, int, float, bool, list, tuple, dict)):
                        log_entry = f"{full_key}: {value}"
                        if logger_handle: logger_handle.info(log_entry)
                        if file_handle: file_handle.write(log_entry + "\n")

            log_namespace(self.config, file_handle=f, logger_handle=self.logger)

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            param_log = f"\nTotal model parameters: {total_params:,}\nTrainable parameters: {trainable_params:,}"
            self.logger.info(param_log)
            f.write(param_log + "\n")
            
            end_marker = "=" * 50
            self.logger.info(end_marker)
            f.write(end_marker + "\n")
            
    def print_metrics(self, epoch, train_metrics, val_metrics):
        self.logger.info(f"\nEpoch {epoch}:")
        self.logger.info("-" * 50)

        self.logger.info("Training Metrics:")
        self.logger.info(f"Total Loss: {train_metrics.get('total_loss', float('nan')):.4f}")
        self.logger.info(f"Fusion Loss: {train_metrics.get('fusion_loss', float('nan')):.4f}")
        self.logger.info(f"Swin Loss: {train_metrics.get('swin_loss', float('nan')):.4f}")
        self.logger.info(f"GRU Loss: {train_metrics.get('gru_loss', float('nan')):.4f}")

        if 'iou' in train_metrics and 'acc' in train_metrics and \
           'precision' in train_metrics and 'recall' in train_metrics and 'f1' in train_metrics:
            self.logger.info("\nPer-Class Metrics:")
            for i, class_name in enumerate(self.class_names):
                if i < len(train_metrics['iou']):
                    self.logger.info(f"\n{class_name}:")
                    self.logger.info(f"  IoU: {train_metrics['iou'][i]:.4f}")
                    self.logger.info(f"  Accuracy: {train_metrics['acc'][i]:.4f}")
                    self.logger.info(f"  Precision: {train_metrics['precision'][i]:.4f}")
                    self.logger.info(f"  Recall: {train_metrics['recall'][i]:.4f}")
                    self.logger.info(f"  F1 Score: {train_metrics['f1'][i]:.4f}")
                else:
                     self.logger.warning(f"Index {i} out of bounds for train class metrics.")
        else:
             self.logger.warning("Missing per-class keys in train metrics.")

        self.logger.info("\nOverall Metrics:")
        self.logger.info(f"Average Accuracy (aAcc): {train_metrics.get('accuracy', float('nan')):.4f}")
        self.logger.info(f"Mean IoU (mIoU): {train_metrics.get('miou', float('nan')):.4f}")
        self.logger.info(f"Mean Class Accuracy (mAcc): {train_metrics.get('macc', float('nan')):.4f}")
        self.logger.info(f"Kappa Coefficient: {train_metrics.get('kappa', float('nan')):.4f}")
        self.logger.info(f"F1 Macro: {train_metrics.get('f1_macro', float('nan')):.4f}")
        self.logger.info(f"F1 Weighted: {train_metrics.get('f1_weighted', float('nan')):.4f}")

        self.logger.info("\nValidation Metrics:")
        self.logger.info(f"Total Loss: {val_metrics.get('total_loss', float('nan')):.4f}")
        self.logger.info(f"Fusion Loss: {val_metrics.get('fusion_loss', float('nan')):.4f}")
        self.logger.info(f"Swin Loss: {val_metrics.get('swin_loss', float('nan')):.4f}")
        self.logger.info(f"GRU Loss: {val_metrics.get('gru_loss', float('nan')):.4f}")

        if 'iou' in val_metrics and 'acc' in val_metrics and \
           'precision' in val_metrics and 'recall' in val_metrics and 'f1' in val_metrics:
            self.logger.info("\nValidation Set Per-Class Metrics:")
            for i, class_name in enumerate(self.class_names):
                if i < len(val_metrics['iou']):
                    self.logger.info(f"\n{class_name}:")
                    self.logger.info(f"  IoU: {val_metrics['iou'][i]:.4f}")
                    self.logger.info(f"  Accuracy: {val_metrics['acc'][i]:.4f}")
                    self.logger.info(f"  Precision: {val_metrics['precision'][i]:.4f}")
                    self.logger.info(f"  Recall: {val_metrics['recall'][i]:.4f}")
                    self.logger.info(f"  F1 Score: {val_metrics['f1'][i]:.4f}")
                else:
                    self.logger.warning(f"Index {i} out of bounds for validation class metrics.")
        else:
             self.logger.warning("Missing per-class keys in validation metrics.")

        self.logger.info("\nValidation Set Overall Metrics:")
        self.logger.info(f"Average Accuracy (aAcc): {val_metrics.get('accuracy', float('nan')):.4f}")
        self.logger.info(f"Mean IoU (mIoU): {val_metrics.get('miou', float('nan')):.4f}")
        self.logger.info(f"Mean Class Accuracy (mAcc): {val_metrics.get('macc', float('nan')):.4f}")
        self.logger.info(f"Kappa Coefficient: {val_metrics.get('kappa', float('nan')):.4f}")
        self.logger.info(f"F1 Macro: {val_metrics.get('f1_macro', float('nan')):.4f}")
        self.logger.info(f"F1 Weighted: {val_metrics.get('f1_weighted', float('nan')):.4f}")
        self.logger.info("-" * 50 + "\n")

    def train(self):
        self.logger.info("Starting training...")
        for epoch in range(self.start_epoch + 1, self.config.training.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            
            val_metrics = self.validate()
            
            self.print_metrics(epoch, train_metrics, val_metrics)

            current_lr_before_step = self.optimizer.param_groups[0]['lr']
            if epoch > self.warmup_epochs:
                 self.scheduler.step()
                 current_lr_after_step = self.optimizer.param_groups[0]['lr']
                 self.logger.info(f"Epoch {epoch} > Warmup. Scheduler stepped. LR: {current_lr_before_step:.8f} -> {current_lr_after_step:.8f}")
            else:
                 self.logger.info(f"Epoch {epoch} <= Warmup. LR at end of epoch: {current_lr_before_step:.8f}")

            metric_name = self.config.metrics.main_metric.lower()
            current_metric = val_metrics.get(metric_name, float('-inf'))

            is_best = current_metric > self.best_metric + self.early_stopping_min_delta
            
            if is_best:
                self.logger.info(f"New best model found! Metric ({metric_name}): {current_metric:.4f} > {self.best_metric:.4f}")
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                self.logger.info(f"Metric ({metric_name}) did not improve ({current_metric:.4f} <= {self.best_metric:.4f}). Patience: {self.patience_counter}/{self.early_stopping_patience}")
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            patience = getattr(self.config.training.early_stopping, 'patience', 15)
            if self.patience_counter >= patience:
                self.logger.info(f'Early stopping triggered after {epoch} epochs due to no improvement for {patience} epochs.')
                break

            self.epochs_completed.append(epoch)
            for loss_type in self.train_loss_history.keys():
                self.train_loss_history[loss_type].append(train_metrics.get(f'{loss_type}_loss', float('nan')))
            for loss_type in self.val_loss_history.keys():
                self.val_loss_history[loss_type].append(val_metrics.get(f'{loss_type}_loss', float('nan')))
            self.val_acc_history.append(val_metrics.get('accuracy', float('nan')))
            self.val_miou_history.append(val_metrics.get('miou', float('nan')))
            
            if self.plot_frequency > 0 and epoch % self.plot_frequency == 0:
                 try:
                     self.logger.info(f"Generating curves at epoch {epoch}...")
                     self.plot_and_save_curves()
                 except Exception as plot_e:
                     self.logger.error(f"Error generating plots at epoch {epoch}: {plot_e}")
            
        self.logger.info("Training loop finished.")
        try:
            self.log_expert_preferences()
        except Exception as pref_e:
             self.logger.error(f"Could not log final expert preferences: {pref_e}")
             
        self.logger.info("Generating FINAL loss and accuracy curves...")
        try:
            self.plot_and_save_curves()
            self.logger.info(f"Final curves saved to {self.checkpoint_dir}")
        except Exception as final_plot_e:
             self.logger.error(f"Error generating final plots: {final_plot_e}")

    def setup_curriculum_learning(self):
        self.curriculum_stages = {
            'easy': (0, 0.3),
            'medium': (0.3, 0.6),
            'hard': (0.6, 1.0)
        }
    def get_curriculum_stage(self, epoch):
        progress = epoch / self.config.training.epochs
        if progress < self.curriculum_stages['easy'][1]:
            return 'easy'
        elif progress < self.curriculum_stages['medium'][1]:
            return 'medium'
        else:
            return 'hard'
    def get_curriculum_weight(self, epoch):
        stage = self.get_curriculum_stage(epoch)
        weights = {
            'easy': 1.0,
            'medium': 0.7,
            'hard': 0.4
        }
        return weights[stage]
    def adjust_learning_rate_curriculum(self, epoch):
        if hasattr(self.config.training, 'curriculum') and self.config.training.curriculum:
            weight = self.get_curriculum_weight(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= weight
                self.logger.info(f"Curriculum: Adjusted LR for group to {param_group['lr']:.8f} (weight: {weight})")

    def log_expert_preferences(self):
        if hasattr(self.model, 'category_gate') and hasattr(self.model.category_gate, 'get_expert_preferences'):
            preferences = self.model.category_gate.get_expert_preferences()
            preferences = preferences.detach().cpu().numpy()
            
            self.logger.info("\nFinal Expert Preferences (Softmax Applied):")
            self.logger.info("Class\tHLS Weight\tCDL Weight")
            for i, class_name in enumerate(self.class_names):
                if i < preferences.shape[0]:
                    self.logger.info(f"{class_name}\t{preferences[i][0]:.4f}\t{preferences[i][1]:.4f}")
                else:
                    self.logger.warning(f"Index {i} out of bounds for expert preferences.")
        else:
             self.logger.warning("Could not find 'category_gate' or 'get_expert_preferences' method in model to log preferences.")

    def plot_and_save_curves(self):
        if not self.epochs_completed:
            self.logger.warning("No epochs completed, cannot plot curves.")
            return

        save_path_loss = os.path.join(self.checkpoint_dir, 'loss_curves.png')
        save_path_acc = os.path.join(self.checkpoint_dir, 'accuracy_miou_curves.png')
        save_path_lr = os.path.join(self.checkpoint_dir, 'lr_curve.png')

        epochs = self.epochs_completed
        
        min_len = len(epochs)
        for key in self.train_loss_history:
             min_len = min(min_len, len(self.train_loss_history[key]))
        for key in self.val_loss_history:
             min_len = min(min_len, len(self.val_loss_history[key]))
        min_len = min(min_len, len(self.val_acc_history), len(self.val_miou_history), len(self.lr_history))

        if min_len < len(epochs):
             self.logger.warning(f"History list lengths ({min_len}) mismatch number of epochs ({len(epochs)}). Truncating plot data.")
             epochs = epochs[:min_len]
             for key in self.train_loss_history:
                 self.train_loss_history[key] = self.train_loss_history[key][:min_len]
             for key in self.val_loss_history:
                 self.val_loss_history[key] = self.val_loss_history[key][:min_len]
             self.val_acc_history = self.val_acc_history[:min_len]
             self.val_miou_history = self.val_miou_history[:min_len]
             self.lr_history = self.lr_history[:min_len]

        def to_cpu_numpy(data_list):
            result = []
            for item in data_list:
                if torch.is_tensor(item):
                    result.append(item.cpu().numpy() if item.device.type == 'cuda' else item.numpy())
                elif isinstance(item, (int, float)):
                    result.append(float(item))
                else:
                    result.append(item)
            return result

        epochs = to_cpu_numpy(epochs)
        for key in self.train_loss_history:
            self.train_loss_history[key] = to_cpu_numpy(self.train_loss_history[key])
        for key in self.val_loss_history:
            self.val_loss_history[key] = to_cpu_numpy(self.val_loss_history[key])
        self.val_acc_history = to_cpu_numpy(self.val_acc_history)
        self.val_miou_history = to_cpu_numpy(self.val_miou_history)
        self.lr_history = to_cpu_numpy(self.lr_history)

        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.train_loss_history['total'], label='Train Total Loss', linestyle='-')
        plt.plot(epochs, self.train_loss_history['fusion'], label='Train Fusion Loss', linestyle=':')
        plt.plot(epochs, self.train_loss_history['swin'], label='Train Swin Loss', linestyle=':')
        plt.plot(epochs, self.train_loss_history['gru'], label='Train GRU Loss', linestyle=':')
        plt.plot(epochs, self.val_loss_history['total'], label='Val Total Loss', linestyle='-', linewidth=2)
        plt.plot(epochs, self.val_loss_history['fusion'], label='Val Fusion Loss', linestyle='--')
        plt.plot(epochs, self.val_loss_history['swin'], label='Val Swin Loss', linestyle='--')
        plt.plot(epochs, self.val_loss_history['gru'], label='Val GRU Loss', linestyle='--')
        plt.title('Training and Validation Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        try: plt.savefig(save_path_loss); self.logger.info(f"Loss curves saved to {save_path_loss}")
        except Exception as e: self.logger.error(f"Failed to save loss curves: {e}")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.val_acc_history, label='Validation Accuracy (aAcc)', marker='o')
        plt.plot(epochs, self.val_miou_history, label='Validation mIoU', marker='x')
        plt.title('Validation Accuracy (aAcc) and mIoU Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        try: plt.savefig(save_path_acc); self.logger.info(f"Accuracy/mIoU curves saved to {save_path_acc}")
        except Exception as e: self.logger.error(f"Failed to save accuracy/mIoU curves: {e}")
        plt.close()

        if self.lr_history:
             plt.figure(figsize=(10, 6))
             plt.plot(epochs, self.lr_history, label='Learning Rate', marker='.')
             plt.title('Learning Rate Schedule')
             plt.xlabel('Epoch')
             plt.ylabel('Learning Rate')
             plt.legend()
             plt.grid(True)
             plt.tight_layout()
             try:
                 plt.savefig(save_path_lr)
                 self.logger.info(f"Learning rate curve saved to {save_path_lr}")
             except Exception as e:
                 self.logger.error(f"Failed to save learning rate curve: {e}")
             plt.close()
        else:
             self.logger.warning("LR history is empty, skipping LR plot generation.")
