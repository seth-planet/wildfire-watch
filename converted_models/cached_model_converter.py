#!/usr/bin/env python3.12
"""
Cached Model Converter for Wildfire Watch

This module extends EnhancedModelConverter with caching capabilities
to significantly reduce conversion times during testing and development.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from convert_model import EnhancedModelConverter, ModelInfo
from model_cache_manager import ModelCacheManager, compute_params_hash

logger = logging.getLogger(__name__)


class CachedModelConverter(EnhancedModelConverter):
    """
    Enhanced Model Converter with caching support
    
    This class extends the base converter to check for cached conversions
    before performing expensive operations.
    """
    
    def __init__(self, *args, use_cache: bool = True, cache_dir: str = "cache", **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = use_cache
        self.cache_manager = ModelCacheManager(cache_dir) if use_cache else None
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        
    def _get_calibration_hash(self) -> str:
        """Get hash of calibration data for cache key"""
        if not self.calibration_data or not Path(self.calibration_data).exists():
            return "no_calib"
            
        # For directories, hash the list of files
        calib_path = Path(self.calibration_data)
        if calib_path.is_dir():
            files = sorted(calib_path.rglob("*"))
            content = "\n".join(f.name for f in files if f.is_file())
            return compute_params_hash({'files': content})
        else:
            # For single file, use its hash
            return compute_params_hash({'file': str(calib_path)})[:8]
    
    def _convert_single_format(self, format_name: str, size: int, 
                              accelerator: str = None) -> Dict[str, Any]:
        """Override to add caching"""
        
        # If caching disabled, use parent implementation
        if not self.use_cache:
            return super()._convert_single_format(format_name, size, accelerator)
        
        # Determine precision based on format and accelerator
        precision = self._get_precision_for_format(format_name, accelerator)
        
        # Generate cache key
        cache_key = self.cache_manager.generate_cache_key(
            model_name=self.model_name,
            size=size,
            format=format_name,
            precision=precision,
            calibration_hash=self._get_calibration_hash() if format_name == 'tflite' else None
        )
        
        # Check cache
        cached_path = self.cache_manager.get_cached_model(cache_key)
        if cached_path:
            self.cache_stats['hits'] += 1
            logger.info(f"Using cached model: {cache_key}")
            
            # Return in expected format
            return {
                'path': cached_path,
                'size': Path(cached_path).stat().st_size,
                'cached': True,
                'cache_key': cache_key
            }
        
        # Cache miss - perform conversion
        self.cache_stats['misses'] += 1
        logger.info(f"Cache miss for {cache_key}, converting...")
        
        start_time = time.time()
        result = super()._convert_single_format(format_name, size, accelerator)
        conversion_time = time.time() - start_time
        
        # Cache successful conversions
        if isinstance(result, dict) and 'error' not in result and 'path' in result:
            try:
                # Cache the converted model
                cache_params = {
                    'format': format_name,
                    'size': size,
                    'precision': precision,
                    'accelerator': accelerator,
                    'model_type': self.model_type,
                    'conversion_time': conversion_time
                }
                
                cached_path = self.cache_manager.cache_model(
                    key=cache_key,
                    model_path=result['path'],
                    params=cache_params
                )
                
                result['cached'] = False
                result['cache_key'] = cache_key
                
            except Exception as e:
                logger.error(f"Failed to cache model: {e}")
                self.cache_stats['errors'] += 1
        
        return result
    
    def _get_precision_for_format(self, format_name: str, accelerator: str = None) -> str:
        """Determine precision based on format and accelerator"""
        if format_name == 'tflite':
            return 'int8' if self.calibration_data else 'fp32'
        elif format_name == 'tensorrt':
            return 'fp16' if accelerator in ['gpu', 'jetson'] else 'fp32'
        elif format_name == 'hailo':
            return 'int8'
        elif format_name == 'openvino':
            return 'fp16'
        else:
            return 'fp32'
    
    def convert_all(self, formats: List[str] = None, sizes: List[int] = None,
                   validate: bool = True) -> Dict[str, Any]:
        """Override to add cache statistics"""
        
        # Reset cache stats for this conversion
        self.cache_stats = {'hits': 0, 'misses': 0, 'errors': 0}
        
        # Call parent implementation
        results = super().convert_all(formats, sizes, validate)
        
        # Add cache statistics to results
        if self.use_cache:
            results['cache_stats'] = self.cache_stats.copy()
            
            # Log cache performance
            total_conversions = self.cache_stats['hits'] + self.cache_stats['misses']
            if total_conversions > 0:
                hit_rate = (self.cache_stats['hits'] / total_conversions) * 100
                logger.info(f"Cache performance: {hit_rate:.1f}% hit rate "
                          f"({self.cache_stats['hits']} hits, "
                          f"{self.cache_stats['misses']} misses)")
        
        return results
    
    def download_and_cache_calibration(self, dataset_name: str = 'default') -> Optional[str]:
        """Download and cache calibration data"""
        if not self.use_cache:
            return super().download_calibration_data(dataset_name)
        
        # Check cache first
        cached_path = self.cache_manager.get_cached_calibration(dataset_name)
        if cached_path:
            logger.info(f"Using cached calibration data: {dataset_name}")
            return cached_path
        
        # Download and cache
        downloaded_path = super().download_calibration_data(dataset_name)
        if downloaded_path:
            cached_path = self.cache_manager.cache_calibration_data(
                dataset_name, downloaded_path
            )
            return cached_path
        
        return None
    
    def clear_cache(self):
        """Clear the model cache"""
        if self.cache_manager:
            self.cache_manager.clear_cache()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics"""
        if not self.cache_manager:
            return {'enabled': False}
        
        stats = self.cache_manager.get_cache_stats()
        stats['enabled'] = True
        stats['current_session'] = self.cache_stats.copy()
        
        return stats


def convert_with_cache(model_path: str, output_dir: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to convert a model with caching enabled
    
    Args:
        model_path: Path to input model
        output_dir: Output directory for converted models
        **kwargs: Additional arguments for converter
        
    Returns:
        Conversion results dictionary
    """
    converter = CachedModelConverter(
        model_path=model_path,
        output_dir=output_dir,
        use_cache=True,
        **kwargs
    )
    
    return converter.convert_all()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert models with caching")
    parser.add_argument("model", help="Path to model file")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--formats", "-f", nargs="+", 
                       default=['onnx', 'tflite'],
                       help="Formats to convert to")
    parser.add_argument("--sizes", "-s", nargs="+", type=int,
                       default=[640],
                       help="Model input sizes")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear cache before conversion")
    
    args = parser.parse_args()
    
    # Create converter
    converter = CachedModelConverter(
        model_path=args.model,
        output_dir=args.output,
        use_cache=not args.no_cache
    )
    
    # Clear cache if requested
    if args.clear_cache:
        converter.clear_cache()
        print("Cache cleared")
    
    # Show cache info
    cache_info = converter.get_cache_info()
    if cache_info['enabled']:
        print(f"Cache enabled: {cache_info['total_entries']} entries, "
              f"{cache_info['total_size_mb']:.1f} MB used")
    
    # Perform conversion
    print(f"Converting {args.model} to {args.formats}")
    results = converter.convert_all(formats=args.formats, sizes=args.sizes)
    
    # Show results
    print("\nConversion complete!")
    if 'cache_stats' in results:
        stats = results['cache_stats']
        print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")