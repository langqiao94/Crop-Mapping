import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
import rasterio

"""Base class for transformations"""
class BaseTransform:
    """Base transform class"""
    def __call__(self, results: Dict) -> Dict:
        return results

"""Load HLS satellite images"""
class LoadHLSImageFromFile(BaseTransform):
    """Load HLS satellite images"""
    def __call__(self, results: Dict) -> Dict:
        # Load HLS image data from file
        hls_path = results['hls_path'].replace('\\', '/').replace('Dataset/Dataset', 'Dataset')
        try:
            with rasterio.open(hls_path) as src:
                hls_data = src.read()  # Read all bands
            # Ensure data shape is [C, H, W], where C=current number of bands
            if len(hls_data.shape) == 3:
                if hls_data.shape[0] != 72:               # 72 is the number of HLS bands
                    hls_data = hls_data.transpose(2, 0, 1)
            results['hls'] = torch.from_numpy(hls_data).float()
        except Exception as e:
            print(f"Error: Failed to load HLS file {hls_path}: {str(e)}")
            return None
        return results

"""Load multi-band TIF files containing historical sequences (e.g., 5-band CDL/hist)"""
class LoadMultiBandHistoryFromFile(BaseTransform):
    """Load multi-band TIF files containing historical sequences (e.g., 5-band CDL/hist)"""
    def __init__(self, expected_bands: int = 5):
        self.expected_bands = expected_bands # Expected number of bands (sequence length)

    def __call__(self, results: Dict) -> Optional[Dict]:
        cdl_path = results.get('cdl_path')
        if cdl_path is None:
            print("Error: 'cdl_path' not found in sample.")
            return None 

        # Correct path
        cdl_path = cdl_path.replace('\\', '/').replace('Dataset/Dataset', 'Dataset')

        try:
            with rasterio.open(cdl_path) as src:
                # Check if number of bands matches expectations
                if src.count != self.expected_bands:
                    print(f"Error: File {cdl_path} contains {src.count} bands, but expected {self.expected_bands}. Skipping sample.")
                    return None
                
                # Read all bands (T, H, W)
                hist_data = src.read().astype(np.int64)
                results['history'] = torch.from_numpy(hist_data) # Already a LongTensor

        except Exception as e:
            print(f"Error: Failed to process historical/CDL file {cdl_path}: {str(e)}")
            return None

        return results

"""Load annotation data"""
class LoadAnnotations(BaseTransform):
    """Load annotation data"""
    def __call__(self, results: Dict) -> Dict:
        # Load annotation data from file
        label_path = results['label_path'].replace('\\', '/').replace('Dataset/Dataset', 'Dataset')
        try:
            with rasterio.open(label_path) as src:
                label_data = src.read(1)  # Read first band as label
            results['label'] = torch.from_numpy(label_data).long()
        except Exception as e:
            print(f"Error: Failed to load annotation file {label_path}: {str(e)}")
            return None
        return results

"""Resize images"""
class Resize(BaseTransform):
    """Resize images"""
    def __init__(self, scale: Union[Tuple[int, int], List[int]], keep_ratio: bool = True):
        self.scale = tuple(scale)
        self.keep_ratio = keep_ratio
    
    def __call__(self, results: Dict) -> Dict:
        hls_target_size = None
        original_hls_shape = None
        if 'hls' in results:
            img = results['hls']
            original_hls_shape = img.shape[-2:]
            h, w = original_hls_shape
            if self.keep_ratio:
                new_h, new_w = self.scale
                scale = min(new_h / h, new_w / w)
                h_new = int(h * scale + 0.5)
                w_new = int(w * scale + 0.5)
                hls_target_size = (h_new, w_new)
                img = F.interpolate(img.unsqueeze(0), size=hls_target_size, mode='bilinear', align_corners=False).squeeze(0)
            else:
                hls_target_size = self.scale
                img = F.interpolate(img.unsqueeze(0), size=hls_target_size, mode='bilinear', align_corners=False).squeeze(0)
            results['hls'] = img

        target_size_for_labels = None
        if hls_target_size is not None:
            target_size_for_labels = hls_target_size
        else:
            key_for_shape = None
            original_label_shape = None
            if 'history' in results:
                key_for_shape = 'history'
                # Shape is (T, H, W), take the last two dimensions
                original_label_shape = results['history'].shape[-2:] 
            elif 'label' in results:
                key_for_shape = 'label'
                original_label_shape = results['label'].shape[-2:]
            
            if original_label_shape is not None:
                h, w = original_label_shape
                if self.keep_ratio:
                    new_h, new_w = self.scale
                    scale_ratio = min(new_h / h, new_w / w)
                    h_new = int(h * scale_ratio + 0.5)
                    w_new = int(w * scale_ratio + 0.5)
                    target_size_for_labels = (h_new, w_new)
                else:
                    target_size_for_labels = self.scale
            else:
                target_size_for_labels = self.scale

        if 'history' in results:
            history = results['history'] # Shape (T, H, W)
            history_for_interp = history.unsqueeze(0).float() # Shape (1, T, H, W)
            
            resized_history = F.interpolate(history_for_interp, size=target_size_for_labels, mode='nearest')
            
            results['history'] = resized_history.squeeze(0).long() # Shape (T, H', W')

        if 'label' in results:
            label = results['label'] # Shape (H, W)
            label_for_interp = label.unsqueeze(0).unsqueeze(0).float()
            
            resized_label = F.interpolate(label_for_interp, size=target_size_for_labels, mode='nearest')
            
            results['label'] = resized_label.squeeze(0).squeeze(0).long() # Shape (H', W')

        return results

"""Random flip"""
class RandomFlip(BaseTransform):
    """Random flip"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, results: Dict) -> Dict:
        if np.random.rand() < self.prob:
            for key in ['hls', 'history', 'label']:
                if key in results:
                    results[key] = torch.flip(results[key], dims=[-1])
        return results

"""Standardization (only HLS)"""
class Normalize(BaseTransform):
    """Normalize (HLS only)"""
    def __init__(self, hls_mean: List[float], hls_std: List[float]):
        # Only initialize HLS mean and standard deviation
        self.hls_mean = torch.tensor(hls_mean).view(-1, 1, 1)
        self.hls_std = torch.tensor(hls_std).view(-1, 1, 1)
        # Remove hist_mean and hist_std
    
    def __call__(self, results: Dict) -> Dict:
        # Normalize HLS images
        if 'hls' in results:
            # Ensure hls data is FloatTensor
            if not isinstance(results['hls'], torch.FloatTensor):
                 # Type conversion if needed, but this is usually done during loading
                 results['hls'] = results['hls'].float() 
            results['hls'] = (results['hls'] - self.hls_mean.to(results['hls'].device)) / self.hls_std.to(results['hls'].device)
        
        # Remove history normalization logic
        # if 'history' in results:
        #     results['history'] = (results['history'] - self.hist_mean) / self.hist_std
        return results

"""Pack segmentation task inputs"""
class PackSegInputs(BaseTransform):
    """Pack segmentation task inputs"""
    def __call__(self, results: Dict) -> Dict:
        # Ensure all tensors are on the correct device and contiguous
        for key in ['hls', 'history', 'label']:
            if key in results:
                results[key] = results[key].contiguous()
        return results

def build_transform(transform_cfg):
    """Build transform class"""
    # If it's a SimpleNamespace object, convert to dictionary
    if hasattr(transform_cfg, '__dict__'):
        transform_cfg = vars(transform_cfg)
    else:
        transform_cfg = transform_cfg.copy()  # Create copy to avoid modifying original config
    
    transform_type = transform_cfg.pop('type')
    if transform_type == 'LoadHLSImageFromFile':
        return LoadHLSImageFromFile()
    elif transform_type == 'LoadMultiBandHistoryFromFile':
        return LoadMultiBandHistoryFromFile(**transform_cfg)
    elif transform_type == 'LoadAnnotations':
        return LoadAnnotations()
    elif transform_type == 'Resize':
        return Resize(**transform_cfg)
    elif transform_type == 'RandomFlip':
        return RandomFlip(**transform_cfg)
    elif transform_type == 'Normalize':
        # Extract only hls-related parameters from configuration
        hls_mean = transform_cfg.get('hls_mean')
        hls_std = transform_cfg.get('hls_std')
        if hls_mean is None or hls_std is None:
            raise ValueError("Normalize transform requires 'hls_mean' and 'hls_std' in config.")
        return Normalize(hls_mean=hls_mean, hls_std=hls_std)
    elif transform_type == 'PackSegInputs':
        return PackSegInputs()
    else:
        raise ValueError(f'Unknown transform type: {transform_type}') 