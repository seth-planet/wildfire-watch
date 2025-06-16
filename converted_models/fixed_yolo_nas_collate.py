#!/usr/bin/env python3.10
"""Fixed collate function for YOLO-NAS dataloaders"""

import torch
from typing import List, Tuple, Any

class FixedCollateFunction:
    """Custom collate function that ensures targets are in the correct format for PPYoloELoss"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        
    def __call__(self, batch: List[Tuple[Any, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate batch data and ensure targets are formatted correctly.
        
        Args:
            batch: List of (image, target) tuples from dataset
            
        Returns:
            images: Tensor of shape [B, 3, H, W]
            targets: Tensor of shape [N, 6] where each row is [image_idx, class_id, cx, cy, w, h]
        """
        images = []
        all_targets = []
        
        for batch_idx, (image, target) in enumerate(batch):
            # Convert numpy array to tensor if needed
            if hasattr(image, 'numpy'):  # It's already a tensor
                # Check if we need to transpose from HWC to CHW
                if image.dim() == 3 and image.shape[-1] == 3:
                    # HWC format, transpose to CHW
                    image = image.permute(2, 0, 1)
                images.append(image)
            else:  # It's a numpy array
                # Convert to tensor
                image_tensor = torch.from_numpy(image)
                # Check if we need to transpose from HWC to CHW
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    # HWC format, transpose to CHW
                    image_tensor = image_tensor.permute(2, 0, 1)
                images.append(image_tensor)
            
            # Ensure all targets are float32
            if isinstance(target, torch.Tensor):
                target = target.float()
            
            # Handle different target formats
            if isinstance(target, torch.Tensor):
                if target.numel() > 0:
                    # Ensure target is 2D
                    if len(target.shape) == 1:
                        target = target.unsqueeze(0)
                    
                    # Check if we need to add image index
                    if target.shape[-1] == 5:
                        # Target format is [class_id, cx, cy, w, h]
                        # Add image index as first column
                        batch_indices = torch.full((target.shape[0], 1), batch_idx, dtype=target.dtype)
                        target = torch.cat([batch_indices, target], dim=1)
                    elif target.shape[-1] == 6:
                        # Target already has image index, update it
                        target[:, 0] = batch_idx
                    else:
                        print(f"Warning: Unexpected target shape {target.shape}")
                        continue
                    
                    # Validate class indices
                    class_indices = target[:, 1].long()
                    valid_mask = (class_indices >= 0) & (class_indices < self.num_classes)
                    
                    if not valid_mask.all():
                        invalid_classes = class_indices[~valid_mask].unique().tolist()
                        print(f"Warning: Invalid class indices {invalid_classes} found, filtering...")
                        target = target[valid_mask]
                    
                    if target.shape[0] > 0:
                        all_targets.append(target)
            
            elif isinstance(target, list) and len(target) > 0:
                # Handle list format (shouldn't happen with standard dataloader)
                print("Warning: Got list format targets, converting...")
                for t in target:
                    if isinstance(t, torch.Tensor) and t.numel() > 0:
                        if len(t.shape) == 1:
                            t = t.unsqueeze(0)
                        if t.shape[-1] == 5:
                            batch_indices = torch.full((t.shape[0], 1), batch_idx, dtype=t.dtype)
                            t = torch.cat([batch_indices, t], dim=1)
                        all_targets.append(t)
        
        # Stack images
        images = torch.stack(images, dim=0)
        
        # Convert images to float32 and normalize to [0, 1]
        # Super-gradients expects float32 tensors normalized to [0, 1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif not images.dtype.is_floating_point:
            images = images.float()
        elif images.max() > 1.0:
            # If float but in [0, 255] range, normalize
            images = images / 255.0
        
        # Concatenate all targets
        if all_targets:
            targets = torch.cat(all_targets, dim=0)
        else:
            # Return empty tensor with correct shape if no targets
            targets = torch.zeros((0, 6), dtype=torch.float32)
        
        return images, targets


def wrap_dataloader_with_fixed_collate(dataloader, num_classes: int):
    """Wrap a dataloader with the fixed collate function"""
    # Create new dataloader with custom collate
    from torch.utils.data import DataLoader
    
    # Handle wrapped dataloaders that may not expose all attributes
    if hasattr(dataloader, 'dataloader'):
        # This is a wrapped dataloader, get the underlying one
        base_loader = dataloader.dataloader
    else:
        base_loader = dataloader
    
    # Get batch size - try multiple methods
    batch_size = None
    if hasattr(base_loader, 'batch_size'):
        batch_size = base_loader.batch_size
    elif hasattr(base_loader, 'batch_sampler') and hasattr(base_loader.batch_sampler, 'batch_size'):
        batch_size = base_loader.batch_sampler.batch_size
    else:
        # Default fallback
        batch_size = 8
    
    # Get dataset
    dataset = base_loader.dataset if hasattr(base_loader, 'dataset') else dataloader.dataset
    
    # Get num_workers
    num_workers = base_loader.num_workers if hasattr(base_loader, 'num_workers') else 0
    
    # Build dataloader kwargs
    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': isinstance(base_loader.sampler, torch.utils.data.RandomSampler) if hasattr(base_loader, 'sampler') else False,
        'num_workers': num_workers,
        'collate_fn': FixedCollateFunction(num_classes),
        'pin_memory': base_loader.pin_memory if hasattr(base_loader, 'pin_memory') else False,
        'drop_last': base_loader.drop_last if hasattr(base_loader, 'drop_last') else False,
        'timeout': base_loader.timeout if hasattr(base_loader, 'timeout') else 0,
        'worker_init_fn': base_loader.worker_init_fn if hasattr(base_loader, 'worker_init_fn') else None,
    }
    
    # Only add prefetch_factor and persistent_workers if num_workers > 0
    if num_workers > 0:
        kwargs['prefetch_factor'] = base_loader.prefetch_factor if hasattr(base_loader, 'prefetch_factor') else 2
        kwargs['persistent_workers'] = base_loader.persistent_workers if hasattr(base_loader, 'persistent_workers') else False
    
    return DataLoader(**kwargs)