import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import rasterio
from typing import List, Dict, Any
from .transforms import build_transform

class CropMappingDataset(Dataset):
    """
    Dataset class for Crop Mapping
    """
    def __init__(self, data_file: str, pipeline: List[Dict], data_root: str):
        """
        Initialize dataset
        Args:
            data_file: Path to data file, each line contains paths for hls, history(cdl), label files
            pipeline: Data processing pipeline configuration
            data_root: Data root directory
        """
        super().__init__()
        self.data_root = data_root  # Save as string
        self.samples = self._load_file_list(data_file)
        self.transforms = [build_transform(t) for t in pipeline]

    def _load_file_list(self, data_file: str) -> List[Dict[str, str]]:
        """Load file list"""
        samples = []
        # Directly use data_file as it is already the correct path relative to the working directory
        with open(data_file, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) == 3:
                    hls_path_str, cdl_path_str, label_path_str = parts
                    
                    # Convert all paths to forward slash format
                    hls_path_str = hls_path_str.replace('\\', '/')
                    cdl_path_str = cdl_path_str.replace('\\', '/')
                    label_path_str = label_path_str.replace('\\', '/')
                    
                    # Check if file exists
                    hls_full_path = os.path.join(self.data_root, hls_path_str)
                    cdl_full_path = os.path.join(self.data_root, cdl_path_str)
                    label_full_path = os.path.join(self.data_root, label_path_str)
                    
                    # Convert Windows path separators to current system separators
                    hls_full_path = os.path.normpath(hls_full_path)
                    cdl_full_path = os.path.normpath(cdl_full_path)
                    label_full_path = os.path.normpath(label_full_path)
                    
                    if not os.path.exists(hls_full_path):
                        print(f"Warning: HLS file not found {hls_full_path} (line {line_num+1}). Skipping sample.")
                        continue
                    if not os.path.exists(cdl_full_path):
                        print(f"Warning: CDL/History file not found {cdl_full_path} (line {line_num+1}). Skipping sample.")
                        continue
                    if not os.path.exists(label_full_path):
                        print(f"Warning: Label file not found {label_full_path} (line {line_num+1}). Skipping sample.")
                        continue
                    
                    samples.append({
                        'hls_path': hls_path_str,
                        'cdl_path': cdl_path_str,
                        'label_path': label_path_str
                    })
                else:
                    print(f"Warning: Skipping incorrectly formatted line (line {line_num+1}): {line.strip()}")
        return samples

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get data sample"""
        sample = self.samples[idx].copy()
        
        # Build complete path and normalize
        sample['hls_path'] = os.path.normpath(os.path.join(self.data_root, sample['hls_path']))
        sample['cdl_path'] = os.path.normpath(os.path.join(self.data_root, sample['cdl_path']))
        sample['label_path'] = os.path.normpath(os.path.join(self.data_root, sample['label_path']))
        
        # Convert path separators to forward slash (required by rasterio)
        sample['hls_path'] = sample['hls_path'].replace('\\', '/')
        sample['cdl_path'] = sample['cdl_path'].replace('\\', '/')
        sample['label_path'] = sample['label_path'].replace('\\', '/')

        # Apply data processing pipeline
        processed_sample = sample
        for t in self.transforms:
            processed_sample = t(processed_sample)
            if processed_sample is None:
                print(f"Warning: Sample {idx} became None after transformation {type(t).__name__}.")
                return None

        return processed_sample

def create_data_loaders(config):
    """
    Create data loaders
    Args:
        config: Configuration object
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create dataset instances
    train_dataset = CropMappingDataset(
        data_file=config.data.train_file,
        pipeline=config.data.train_pipeline,
        data_root=config.data.data_root
    )
    val_dataset = CropMappingDataset(
        data_file=config.data.val_file,
        pipeline=config.data.val_pipeline,
        data_root=config.data.data_root
    )
    test_dataset = CropMappingDataset(
        data_file=config.data.test_file,
        pipeline=config.data.test_pipeline,
        data_root=config.data.data_root
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        # Consider adding custom collate_fn to handle samples that return None
        # collate_fn=lambda batch: [item for item in batch if item is not None]
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        # collate_fn=lambda batch: [item for item in batch if item is not None]
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        # collate_fn=lambda batch: [item for item in batch if item is not None]
    )
    
    return train_loader, val_loader, test_loader 