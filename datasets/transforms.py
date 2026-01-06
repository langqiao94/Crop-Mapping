import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
import rasterio

class BaseTransform:
    def __call__(self, results: Dict) -> Dict:
        return results

class LoadHLSImageFromFile(BaseTransform):
    def __call__(self, results: Dict) -> Dict:
        hls_path = results['hls_path'].replace('\\', '/').replace('Dataset/Dataset', 'Dataset')
        try:
            with rasterio.open(hls_path) as src:
                hls_data = src.read()
            if len(hls_data.shape) == 3:
                if hls_data.shape[0] != 72:
                    hls_data = hls_data.transpose(2, 0, 1)
            results['hls'] = torch.from_numpy(hls_data).float()
        except Exception as e:
            print(f"Error: Failed to load HLS file {hls_path}: {str(e)}")
            return None
        return results

class LoadMultiBandHistoryFromFile(BaseTransform):
    def __init__(self, expected_bands: int = 5):
        self.expected_bands = expected_bands

    def __call__(self, results: Dict) -> Optional[Dict]:
        cdl_path = results.get('cdl_path')
        if cdl_path is None:
            print("Error: 'cdl_path' not found in sample.")
            return None 

        cdl_path = cdl_path.replace('\\', '/').replace('Dataset/Dataset', 'Dataset')

        try:
            with rasterio.open(cdl_path) as src:
                if src.count != self.expected_bands:
                    print(f"Error: File {cdl_path} contains {src.count} bands, but expected {self.expected_bands}. Skipping sample.")
                    return None
                
                hist_data = src.read().astype(np.int64)
                results['history'] = torch.from_numpy(hist_data)

        except Exception as e:
            print(f"Error: Failed to process historical/CDL file {cdl_path}: {str(e)}")
            return None

        return results

class LoadAnnotations(BaseTransform):
    def __call__(self, results: Dict) -> Dict:
        label_path = results['label_path'].replace('\\', '/').replace('Dataset/Dataset', 'Dataset')
        try:
            with rasterio.open(label_path) as src:
                label_data = src.read(1)
            results['label'] = torch.from_numpy(label_data).long()
        except Exception as e:
            print(f"Error: Failed to load annotation file {label_path}: {str(e)}")
            return None
        return results

class Resize(BaseTransform):
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
            history = results['history']
            history_for_interp = history.unsqueeze(0).float()
            
            resized_history = F.interpolate(history_for_interp, size=target_size_for_labels, mode='nearest')
            
            results['history'] = resized_history.squeeze(0).long()

        if 'label' in results:
            label = results['label']
            label_for_interp = label.unsqueeze(0).unsqueeze(0).float()
            
            resized_label = F.interpolate(label_for_interp, size=target_size_for_labels, mode='nearest')
            
            results['label'] = resized_label.squeeze(0).squeeze(0).long()

        return results

class RandomFlip(BaseTransform):
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, results: Dict) -> Dict:
        if np.random.rand() < self.prob:
            for key in ['hls', 'history', 'label']:
                if key in results:
                    results[key] = torch.flip(results[key], dims=[-1])
        return results

class Normalize(BaseTransform):
    def __init__(self, hls_mean: List[float], hls_std: List[float]):
        self.hls_mean = torch.tensor(hls_mean).view(-1, 1, 1)
        self.hls_std = torch.tensor(hls_std).view(-1, 1, 1)
    
    def __call__(self, results: Dict) -> Dict:
        if 'hls' in results:
            if not isinstance(results['hls'], torch.FloatTensor):
                 results['hls'] = results['hls'].float()
            results['hls'] = (results['hls'] - self.hls_mean.to(results['hls'].device)) / self.hls_std.to(results['hls'].device)
        return results

class PackSegInputs(BaseTransform):
    def __call__(self, results: Dict) -> Dict:
        for key in ['hls', 'history', 'label']:
            if key in results:
                results[key] = results[key].contiguous()
        return results

def build_transform(transform_cfg):
    if hasattr(transform_cfg, '__dict__'):
        transform_cfg = vars(transform_cfg)
    else:
        transform_cfg = transform_cfg.copy()
    
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
        hls_mean = transform_cfg.get('hls_mean')
        hls_std = transform_cfg.get('hls_std')
        if hls_mean is None or hls_std is None:
            raise ValueError("Normalize transform requires 'hls_mean' and 'hls_std' in config.")
        return Normalize(hls_mean=hls_mean, hls_std=hls_std)
    elif transform_type == 'PackSegInputs':
        return PackSegInputs()
    else:
        raise ValueError(f'Unknown transform type: {transform_type}')
