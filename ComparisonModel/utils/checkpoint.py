import os
import torch
from pathlib import Path


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth', best_filename='model_best.pth', save_last_only=False):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if save_last_only:
        filepath = checkpoint_dir / filename
        torch.save(state, filepath)
        print(f"Last checkpoint saved to {filepath}")
    else:
        filepath = checkpoint_dir / 'checkpoint.pth'
        torch.save(state, filepath)
        print(f"Checkpoint saved to {filepath}")
        
        if is_best:
            best_filepath = checkpoint_dir / best_filename
            torch.save(state, best_filepath)
            print(f"Best model saved to {best_filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"Model weights loaded from {checkpoint_path}")
    except RuntimeError as e:
        print(f"Warning: Strict loading failed: {e}")
        print(f"Attempting non-strict loading (some weights may not be loaded)...")
        
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        if missing_keys:
            print(f"Missing keys (not loaded): {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"     - {key}")
            else:
                for key in missing_keys[:10]:
                    print(f"     - {key}")
                print(f"     ... and {len(missing_keys) - 10} more")
        
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"     - {key}")
            else:
                for key in unexpected_keys[:10]:
                    print(f"     - {key}")
                print(f"     ... and {len(unexpected_keys) - 10} more")
        
        if missing_keys and len(missing_keys) > len(model.state_dict()) * 0.5:
            print(f"\nERROR: Checkpoint appears to be incompatible with current model architecture!")
            print(f"   Missing {len(missing_keys)} out of {len(model.state_dict())} model parameters.")
            print(f"   This checkpoint was likely saved with a different model architecture.")
            print(f"   Please train from scratch or use a compatible checkpoint.")
            raise RuntimeError(
                f"Checkpoint incompatible: {len(missing_keys)}/{len(model.state_dict())} parameters missing. "
                f"This checkpoint was saved with a different model architecture. "
                f"Please train from scratch."
            )
        else:
            print(f"Model weights partially loaded from {checkpoint_path} (non-strict mode)")
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer state loaded")
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Scheduler state loaded")
    
    return checkpoint
