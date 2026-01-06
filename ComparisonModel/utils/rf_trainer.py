import os
import time
import yaml
import torch
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

from .metrics import MetricsCalculator, print_metrics, save_metrics_to_file
from .data_loader import get_dataloaders


class RandomForestTrainer:
    def __init__(self, model, config, model_name='random_forest', device='cpu', save_name=None):
        self.model = model
        self.config = config
        self.model_name = model_name
        self.device = 'cpu'
        self.save_name = save_name
        
        print("\nCreating dataloaders...")
        dataset_config_path = config.get('dataset_config', 'config/dataset_config.yaml')
        
        with open(dataset_config_path, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
        
        self.image_root = dataset_config['dataset']['image_root']
        self.label_root = dataset_config['dataset']['label_root']
        
        batch_size = config.get('batch_size', 24)
        file_list_override = config.get('file_list_override', None)
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            dataset_config_path,
            batch_size=batch_size,
            num_workers=config.get('num_workers', 4),
            file_list_override=file_list_override
        )
        
        num_classes = config.get('num_classes', 6)
        class_names = config.get('class_names', None)
        self.metrics_calc = MetricsCalculator(num_classes=num_classes, class_names=class_names)
        
        self.checkpoint_dir = Path(config.get('checkpoint_dir', f'checkpoints/{model_name}'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.get('log_dir', f'logs/{model_name}'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.checkpoint_dir / 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        print(f"\n{'='*80}")
        print(f"RandomForest Trainer initialized for {model_name}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"Logs: {self.log_dir}")
        print(f"{'='*80}\n")
    
    def train(self):
        print("\n" + "="*80)
        print("Training Random Forest Model")
        print("="*80)
        
        start_time = time.time()
        
        self.model.fit(self.train_loader, device=self.device)
        
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        
        print("\nValidating on validation set...")
        val_metrics = self.validate()
        
        print("\n" + "="*80)
        print("Validation Results:")
        print("="*80)
        print(f"Accuracy: {val_metrics['accuracy']:.2f}%")
        print(f"mIoU: {val_metrics['miou']:.2f}%")
        print(f"mACC: {val_metrics['macc']:.2f}%")
        print(f"Kappa: {val_metrics['kappa']:.4f}")
        print(f"F1 Macro: {val_metrics['f1_macro']:.2f}%")
        print(f"F1 Weighted: {val_metrics['f1_weighted']:.2f}%")
        
        model_filename = f'model_best_{self.save_name}.pkl' if self.save_name else 'model_best.pkl'
        model_path = self.checkpoint_dir / model_filename
        self.model.save(model_path)
        
        print(f"\nModel saved to {model_path}")
        
        self._save_feature_importances()
        
        return val_metrics
    
    def validate(self):
        self.metrics_calc.reset()
        
        print("Running validation...")
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                if len(batch) == 4:
                    images, labels, _, _ = batch
                else:
                    images, labels = batch
                
                predictions = self.model.predict(images)
                predictions = torch.from_numpy(predictions)
                
                B = labels.shape[0]
                for i in range(B):
                    pred_flat = predictions[i].flatten()
                    label_flat = labels[i].flatten()
                    self.metrics_calc.update(pred_flat.numpy(), label_flat.numpy())
        
        metrics = self.metrics_calc.get_metrics()
        
        return metrics
    
    def test(self, test_file=None, save_name=None):
        print("\n" + "="*80)
        print("Testing Random Forest Model")
        print("="*80)
        
        if save_name:
            results_dir = Path(f'results/{self.model_name}/{save_name}')
        elif self.save_name:
            results_dir = Path(f'results/{self.model_name}/{self.save_name}')
        else:
            results_dir = Path(f'results/{self.model_name}')
        
        results_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = results_dir / 'predictions'
        pred_dir.mkdir(exist_ok=True)
        
        self.metrics_calc.reset()
        
        print(f"\nRunning inference on test set...")
        print(f"Saving results to: {results_dir}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                if len(batch) == 4:
                    images, labels, img_paths, label_paths = batch
                else:
                    images, labels = batch
                    img_paths = [None] * images.shape[0]
                    label_paths = [None] * images.shape[0]
                
                predictions = self.model.predict(images)
                
                B = images.shape[0]
                for i in range(B):
                    pred = predictions[i]
                    label = labels[i].numpy()
                    
                    self.metrics_calc.update(pred.flatten(), label.flatten())
                    
                    if img_paths[i] is not None:
                        img_path = img_paths[i]
                        
                        full_img_path = os.path.join(self.image_root, img_path)
                        
                        img_filename = Path(img_path).stem
                        output_filename = f"{img_filename}_pred.tif"
                        output_path = pred_dir / output_filename
                        
                        try:
                            with rasterio.open(full_img_path) as src:
                                profile = src.profile.copy()
                                profile.update({
                                    'count': 1,
                                    'dtype': 'uint8',
                                    'compress': 'lzw'
                                })
                                
                                with rasterio.open(output_path, 'w', **profile) as dst:
                                    dst.write(pred.astype(np.uint8), 1)
                        except Exception as e:
                            print(f"\nWarning: Could not save prediction for {Path(img_path).name}: {e}")
        
        print("\n" + "="*80)
        print("Test Results:")
        print("="*80)
        
        metrics = self.metrics_calc.get_metrics()
        
        print_metrics(metrics, self.metrics_calc.class_names)
        
        metrics_file = results_dir / 'metrics.txt'
        save_metrics_to_file(metrics, metrics_file, self.metrics_calc.class_names)
        
        print(f"\n{'='*80}")
        print(f"Results saved to {results_dir}")
        print(f"  - Metrics: {metrics_file}")
        print(f"  - Predictions: {pred_dir}/ ({len(list(pred_dir.glob('*.tif')))} TIF files)")
        print(f"{'='*80}\n")
        
        return metrics
    
    def _save_feature_importances(self):
        importances = self.model.get_feature_importances()
        
        importance_file = self.checkpoint_dir / 'feature_importances.txt'
        
        with open(importance_file, 'w', encoding='utf-8') as f:
            f.write("Feature Importances (Random Forest)\n")
            f.write("="*80 + "\n\n")
            
            sorted_indices = np.argsort(importances)[::-1]
            
            f.write(f"Top 50 Most Important Features:\n")
            f.write("-"*80 + "\n")
            
            for rank, idx in enumerate(sorted_indices[:50], 1):
                f.write(f"{rank:3d}. Feature {idx:4d}: {importances[idx]:.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Total features: {len(importances)}\n")
            f.write(f"Sum of importances: {importances.sum():.6f}\n")
        
        print(f"\nFeature importances saved to {importance_file}")
