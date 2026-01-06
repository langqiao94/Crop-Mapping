import os
import yaml
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class CropMappingDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        label_root: str,
        file_list: str,
        normalize: bool = True,
        mean: list = None,
        std: list = None,
        augment: bool = False,
        img_size: int = 128
    ):
        super().__init__()
        
        self.image_root = image_root
        self.label_root = label_root
        self.normalize = normalize
        self.augment = augment
        self.img_size = img_size
        
        with open(file_list, 'r') as f:
            lines = f.read().splitlines()
        
        self.file_pairs = []
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path, label_path = parts[0], parts[1]
                    img_path = img_path.replace('\\', '/')
                    label_path = label_path.replace('\\', '/')
                    self.file_pairs.append((img_path, label_path))
        
        print(f"Loaded {len(self.file_pairs)} samples from {file_list}")
        
        if self.normalize:
            self.mean = mean if mean is not None else []
            self.std = std if std is not None else []
    
    def __len__(self):
        return len(self.file_pairs)
    
    def load_image(self, image_path: str) -> np.ndarray:
        full_path = os.path.join(self.image_root, image_path)
        full_path = os.path.normpath(full_path)
        
        with rasterio.open(full_path) as src:
            image = src.read()
            
            nodata = src.nodata
            if nodata is not None:
                image = np.where(image == nodata, 0, image)
            
            return image.astype(np.float32)
    
    def load_label(self, label_path: str) -> np.ndarray:
        full_path = os.path.join(self.label_root, label_path)
        full_path = os.path.normpath(full_path)
        
        with rasterio.open(full_path) as src:
            label = src.read(1)
            
            nodata = src.nodata
            if nodata is not None:
                label = np.where(label == nodata, 0, label)
            
            return label.astype(np.int64)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        if len(self.mean) == 0 or len(self.std) == 0:
            image = image / 10000.0
            return np.clip(image, 0, 1)
        else:
            mean = np.array(self.mean, dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(self.std, dtype=np.float32).reshape(-1, 1, 1)
            image = (image - mean) / (std + 1e-8)
            return image
    
    def augment_data(self, image: np.ndarray, label: np.ndarray):
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=1).copy()
        
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()
        
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            label = np.rot90(label, k=k, axes=(0, 1)).copy()
        
        return image, label
    
    def __getitem__(self, idx: int):
        img_path, label_path = self.file_pairs[idx]
        
        image = self.load_image(img_path)
        label = self.load_label(label_path)
        
        if self.normalize:
            image = self.normalize_image(image)
        
        if self.augment:
            image, label = self.augment_data(image, label)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        return image, label, img_path, label_path


def get_dataloaders(config_path: str, batch_size: int = None, num_workers: int = 4, 
                    file_list_override: dict = None):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    dataset_cfg = config.get('dataset', {})
    data_info = config.get('data_info', {})
    dataloader_cfg = config.get('dataloader', {})
    
    if batch_size is None:
        batch_size = dataloader_cfg.get('train', {}).get('batch_size', 24)
    
    image_root = dataset_cfg.get('image_root', 'Dataset/HLS/72')
    label_root = dataset_cfg.get('label_root', 'Dataset/Labels')
    train_file = dataset_cfg.get('train_file_list', 'Dataset/202201_train.txt')
    val_file = dataset_cfg.get('val_file_list', 'Dataset/202201_val.txt')
    test_file = dataset_cfg.get('test_file_list', 'Dataset/202201_test.txt')
    
    if file_list_override:
        if file_list_override.get('train'):
            train_file = file_list_override['train']
        if file_list_override.get('val'):
            val_file = file_list_override['val']
        if file_list_override.get('test'):
            test_file = file_list_override['test']
    
    img_size = data_info.get('image_size', 128)
    
    norm_cfg = config.get('normalization', {})
    normalize_enabled = norm_cfg.get('enabled', True)
    
    train_dataset = CropMappingDataset(
        image_root=image_root,
        label_root=label_root,
        file_list=train_file,
        normalize=normalize_enabled,
        augment=True,
        img_size=img_size
    )
    
    val_dataset = CropMappingDataset(
        image_root=image_root,
        label_root=label_root,
        file_list=val_file,
        normalize=normalize_enabled,
        augment=False,
        img_size=img_size
    )
    
    test_dataset = CropMappingDataset(
        image_root=image_root,
        label_root=label_root,
        file_list=test_file,
        normalize=normalize_enabled,
        augment=False,
        img_size=img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing data loader...")
    train_loader, val_loader, test_loader = get_dataloaders('../config/dataset_config.yaml')
    
    for images, labels in train_loader:
        print(f"Batch - Images: {images.shape}, Labels: {labels.shape}")
        break
    
    print("Data loader test passed!")
