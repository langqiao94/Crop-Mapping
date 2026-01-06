import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace
from datasets.dataloader import CropMappingDataset
from models.fusion_model import ExpertFusionModel
from utils.trainer import Trainer
from utils.logger import setup_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def dict_to_namespace(d):
    if isinstance(d, dict):
        namespace = {}
        for key, value in d.items():
            if isinstance(value, dict):
                namespace[key] = dict_to_namespace(value)
            elif isinstance(value, list):
                namespace[key] = [dict_to_namespace(item) if isinstance(item, dict) else item for item in value]
            else:
                namespace[key] = value
        return SimpleNamespace(**namespace)
    elif isinstance(d, list):
        return [dict_to_namespace(item) if isinstance(item, dict) else item for item in d]
    else:
        return d

def load_config():
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    with open(config_dict['data_config'], 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    config_dict['dataset'] = dataset_config
    
    config = dict_to_namespace(config_dict)
    
    if not hasattr(config, 'dataset'):
        raise ValueError("Missing dataset section in configuration")
    if not hasattr(config.dataset, 'pipeline'):
        raise ValueError("Missing dataset.pipeline section in configuration")
    
    return config

def main():
    logger = setup_logger()
    
    config = load_config()
    logger.info("Configuration files loaded successfully")
    
    train_dataset = CropMappingDataset(
        data_file=os.path.join(config.dataset.data_root, config.dataset.data_files.train),
        pipeline=config.dataset.pipeline.train,
        data_root=config.dataset.data_root
    )
    
    val_dataset = CropMappingDataset(
        data_file=os.path.join(config.dataset.data_root, config.dataset.data_files.val),
        pipeline=config.dataset.pipeline.val,
        data_root=config.dataset.data_root
    )
    
    train_loader_config = vars(config.dataset.dataloader.train)
    val_loader_config = vars(config.dataset.dataloader.val)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        **train_loader_config
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        **val_loader_config
    )
    
    logger.info("Data loaders created successfully")
    
    model = ExpertFusionModel(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f"Model created successfully, using device: {device}")
    
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    logger.info("Starting training...")
    
    trainer.train()
    
    logger.info("Training completed")

if __name__ == '__main__':
    main()
