import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
from datasets.dataloader import CropMappingDataset
from models.fusion_model import ExpertFusionModel
from utils.metrics import MetricsCalculator
from utils.logger import setup_logger
import time
import rasterio
import logging

def dict_to_namespace(d):
    namespace = {}
    for key, value in d.items():
        if key == 'pipeline':
            namespace[key] = value
        elif isinstance(value, dict):
            namespace[key] = dict_to_namespace(value)
        elif isinstance(value, list):
            if key == 'train' or key == 'val' or key == 'test':
                namespace[key] = value
            else:
                namespace[key] = [dict_to_namespace(item) if isinstance(item, dict) else item for item in value]
        else:
            namespace[key] = value
    return SimpleNamespace(**namespace)

def load_config():
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    with open(config_dict['data_config'], 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    config_dict['dataset'] = dataset_config
    
    return dict_to_namespace(config_dict)

def save_prediction(pred, save_path, profile):
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    
    profile.update({
        'count': 1,
        'dtype': 'uint8',
        'compress': 'lzw'
    })
    
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(pred.astype('uint8'), 1)

def test(model, test_loader, device, logger, save_dir, config):
    model.eval()
    metrics_calculator = MetricsCalculator()
    
    pred_save_dir = os.path.join(save_dir, 'predictions')
    os.makedirs(pred_save_dir, exist_ok=True)
    
    try:
        num_classes = config.dataset.data_info.num_classes
        class_names = config.dataset.data_info.class_names
        if num_classes is None or class_names is None or len(class_names) != num_classes:
            raise ValueError("Class information missing or inconsistent in config.")
        logger.info(f"Found {num_classes} classes: {class_names}")
    except AttributeError:
        logger.error("Error: Could not find num_classes or class_names in config.dataset.data_info.")
        logger.warning("Using default class info. Please check your config!")
        num_classes = 6
        class_names = ['other', 'corn', 'cotton', 'soybeans', 'spring wheat', 'winter wheat']

    total_weights_per_class = torch.zeros(num_classes, 2, device=device)
    pixel_counts_per_class = torch.zeros(num_classes, device=device)

    logger.info("Starting test...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Test Progress")):
            model_input_batch = {}
            labels = None
            try:
                model_input_batch['hls'] = batch['hls'].to(device)
                model_input_batch['history'] = batch['history'].to(device)
                if 'label' in batch:
                     labels = batch['label'].to(device)
                if 'label_path' not in batch:
                    logger.warning(f"Test Batch {batch_idx}: Missing key 'label_path'. Cannot save individual predictions.")
            except KeyError as e:
                 logger.error(f"Test Batch {batch_idx}: Missing required key: {e}. Available keys: {list(batch.keys())}. Skipping batch.")
                 continue
            except Exception as e:
                 logger.error(f"Error processing test batch {batch_idx} data: {e}. Skipping batch.")
                 continue

            try:
                outputs = model(model_input_batch)
            except Exception as e:
                 logger.error(f"Error during model forward pass in test batch {batch_idx}: {e}. Skipping batch.")
                 continue

            if not isinstance(outputs, dict) or 'output' not in outputs or 'dynamic_weights' not in outputs:
                 logger.error(f"Skipping test batch {batch_idx} due to invalid model output format (missing 'output' or 'dynamic_weights').")
                 continue

            predictions = outputs['output']
            dynamic_weights = outputs['dynamic_weights']

            pred_labels = torch.argmax(predictions, dim=1)
            pred_labels_flat = pred_labels.reshape(-1)

            for c in range(num_classes):
                mask = (pred_labels_flat == c)
                count = mask.sum().item()

                if count > 0:
                    weights_for_class_c = dynamic_weights[mask, c, :]
                    total_weights_per_class[c] += weights_for_class_c.sum(dim=0)
                    pixel_counts_per_class[c] += count

            if labels is not None:
                try:
                    metrics_calculator.update(pred_labels, labels)
                except Exception as e:
                    logger.error(f"Error in metrics update for batch {batch_idx}: {str(e)}")

            if 'label_path' in batch:
                for i in range(pred_labels.shape[0]):
                    try:
                        label_path = batch['label_path'][i]
                        orig_filename = os.path.basename(label_path)
                        filename_without_ext = os.path.splitext(orig_filename)[0]
                        save_name = f"{filename_without_ext}_pred.tif"
                        save_path = os.path.join(pred_save_dir, save_name)

                        with rasterio.open(label_path) as src:
                            profile = src.profile.copy()

                        profile.update({
                            'count': 1,
                            'dtype': 'uint8',
                            'compress': 'lzw'
                        })

                        save_prediction(pred_labels[i], save_path, profile)
                    except Exception as e:
                         logger.error(f"Error saving prediction for sample {i} in batch {batch_idx} (label: {label_path}): {e}")

    metrics = metrics_calculator.get_metrics()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    metrics_file = os.path.join(save_dir, f'metrics_{timestamp}.txt')

    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("Overall Metrics:\n")
        f.write(f"Average Accuracy (aAcc): {metrics.get('accuracy', float('nan')):.4f}\n")
        f.write(f"Mean IoU (mIoU): {metrics.get('miou', float('nan')):.4f}\n")
        f.write(f"Mean Accuracy (mAcc): {metrics.get('macc', float('nan')):.4f}\n")
        f.write(f"Kappa: {metrics.get('kappa', float('nan')):.4f}\n")
        f.write(f"F1 Macro: {metrics.get('f1_macro', float('nan')):.4f}\n")
        f.write(f"F1 Weighted: {metrics.get('f1_weighted', float('nan')):.4f}\n\n")
        
        f.write("Per Class Metrics:\n\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            iou_val = metrics.get('iou', [float('nan')] * num_classes)[i]
            acc_val = metrics.get('acc', [float('nan')] * num_classes)[i]
            pre_val = metrics.get('precision', [float('nan')] * num_classes)[i]
            rec_val = metrics.get('recall', [float('nan')] * num_classes)[i]
            f1_val = metrics.get('f1', [float('nan')] * num_classes)[i]

            f.write(f"  IoU: {iou_val:.4f}\n")
            f.write(f"  Accuracy: {acc_val:.4f}\n")
            f.write(f"  Precision: {pre_val:.4f}\n")
            f.write(f"  Recall: {rec_val:.4f}\n")
            f.write(f"  F1: {f1_val:.4f}\n\n")
    
    logger.info("\nTest Results:")
    logger.info("--------------------------------------------------")
    
    logger.info("Overall Metrics:")
    logger.info(f"Average Accuracy (aAcc): {metrics.get('accuracy', float('nan')):.4f}")
    logger.info(f"Mean IoU (mIoU): {metrics.get('miou', float('nan')):.4f}")
    logger.info(f"Mean Accuracy (mAcc): {metrics.get('macc', float('nan')):.4f}")
    logger.info(f"Kappa: {metrics.get('kappa', float('nan')):.4f}")
    logger.info(f"F1 Macro: {metrics.get('f1_macro', float('nan')):.4f}")
    logger.info(f"F1 Weighted: {metrics.get('f1_weighted', float('nan')):.4f}\n")
    
    logger.info("Per-Class Metrics:")
    for i, class_name in enumerate(class_names):
        iou_val = metrics.get('iou', [float('nan')] * num_classes)[i]
        acc_val = metrics.get('acc', [float('nan')] * num_classes)[i]
        pre_val = metrics.get('precision', [float('nan')] * num_classes)[i]
        rec_val = metrics.get('recall', [float('nan')] * num_classes)[i]
        f1_val = metrics.get('f1', [float('nan')] * num_classes)[i]

        logger.info(f"\n{class_name}:")
        logger.info(f"  IoU: {iou_val:.4f}")
        logger.info(f"  Accuracy: {acc_val:.4f}")
        logger.info(f"  Precision: {pre_val:.4f}")
        logger.info(f"  Recall: {rec_val:.4f}")
        logger.info(f"  F1: {f1_val:.4f}")
    
    logger.info("\n--------------------------------------------------")
    logger.info(f"\nPrediction results saved to: {pred_save_dir}")
    logger.info(f"Evaluation metrics saved to: {metrics_file}")

    logger.info("\n" + "="*20 + " Average Dynamic Gate Weights per Predicted Class (Test Set) " + "="*20)

    header = f"{'Class':<15} | {'Avg Swin Weight':<15} | {'Avg GRU Weight':<15} | {'Pixel Count':<15}"
    logger.info(header)

    separator = "-" * (15 + 3 + 15 + 3 + 15 + 3 + 15)
    logger.info(separator)

    average_weights_per_class = total_weights_per_class / (pixel_counts_per_class.unsqueeze(1) + 1e-8)

    for c in range(num_classes):
        class_name = class_names[c] if c < len(class_names) else f"Class {c}"
        swin_weight = average_weights_per_class[c, 0].item()
        gru_weight = average_weights_per_class[c, 1].item()
        pixel_count = int(pixel_counts_per_class[c].item())

        output_line = f"{class_name:<15} | {swin_weight:<15.4f} | {gru_weight:<15.4f} | {pixel_count:<15}"
        logger.info(output_line)

    logger.info("=" * len(separator))

    log_file = 'training.log'
    has_training_log_handler = False
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.basename(handler.baseFilename) == log_file:
            has_training_log_handler = True
            break
    if has_training_log_handler:
        logger.info(f"Average gate weights also logged to {log_file}")
    else:
        logger.warning(f"Logger does not seem to have a file handler for '{log_file}'. Weights might only be in console/other logs.")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    
    logger = setup_logger()
    
    config = load_config()
    logger.info("Configuration files loaded successfully")
    
    test_dataset = CropMappingDataset(
        data_file=config.dataset.data_files.test,
        pipeline=config.dataset.pipeline['test'],
        data_root=config.dataset.data_root
    )
    
    test_loader_config = vars(config.dataset.dataloader.test)
    test_loader = DataLoader(
        dataset=test_dataset,
        **test_loader_config
    )
    
    logger.info("Test data loader created successfully")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    try:
        model = ExpertFusionModel(config).to(device)
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return
    
    checkpoint_path = os.path.join(config.training.checkpoint.save_dir, 'last_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Successfully loaded model weights: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
            return
    else:
        logger.error(f"Model weight file not found: {checkpoint_path}")
        return
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', f'test_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        test(model, test_loader, device, logger, save_dir, config)
    except Exception as e:
        logger.error(f"An error occurred during the testing process: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()
