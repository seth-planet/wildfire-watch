#!/usr/bin/env python3.12
"""
Model Cache Manager for Wildfire Watch

This module provides caching functionality for converted models to avoid
repeated conversions and reduce test execution time.

Key Features:
- Cache converted models with metadata
- Validate cache entries before use
- Automatic cache cleanup and size management
- Thread-safe operations
"""

import os
import json
import time
import shutil
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached model entry"""
    key: str
    path: str
    timestamp: float
    size_bytes: int
    model_hash: str
    params: Dict[str, Any]
    converter_version: str = "1.0.0"
    
    def is_valid(self, ttl_days: int = 30) -> bool:
        """Check if cache entry is still valid"""
        # Check if file exists
        if not Path(self.path).exists():
            return False
            
        # Check age
        age_days = (time.time() - self.timestamp) / (24 * 3600)
        if age_days > ttl_days:
            return False
            
        # Verify file hash
        current_hash = compute_file_hash(self.path)
        if current_hash != self.model_hash:
            return False
            
        return True


def compute_file_hash(filepath: str, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_params_hash(params: Dict[str, Any]) -> str:
    """Compute hash of conversion parameters"""
    # Sort params for consistent hashing
    sorted_params = json.dumps(params, sort_keys=True)
    return hashlib.md5(sorted_params.encode()).hexdigest()


class ModelCacheManager:
    """
    Manages cached converted models to avoid repeated conversions
    """
    
    def __init__(self, cache_dir: str = "cache", max_size_gb: float = 50, ttl_days: int = 30):
        self.cache_dir = Path(cache_dir)
        self.models_dir = self.cache_dir / "models"
        self.calibration_dir = self.cache_dir / "calibration"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.ttl_days = ttl_days
        self._lock = threading.Lock()
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, CacheEntry]:
        """Load cache metadata from disk"""
        if not self.metadata_file.exists():
            return {}
            
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                
            # Convert to CacheEntry objects
            entries = {}
            for key, entry_data in data.items():
                entries[key] = CacheEntry(**entry_data)
            return entries
            
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
            return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        with self._lock:
            data = {}
            for key, entry in self.metadata.items():
                data[key] = asdict(entry)
                
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    def generate_cache_key(self, model_name: str, size: int, format: str, 
                          precision: str, calibration_hash: Optional[str] = None) -> str:
        """Generate unique cache key for a converted model"""
        components = [
            model_name.replace('/', '_'),  # Sanitize model name
            f"{size}x{size}",
            format,
            precision
        ]
        
        if calibration_hash:
            components.append(calibration_hash[:8])
            
        return "_".join(components)
    
    def get_cached_model(self, key: str) -> Optional[str]:
        """Retrieve cached model path if valid"""
        with self._lock:
            if key not in self.metadata:
                return None
                
            entry = self.metadata[key]
            
            # Validate entry
            if not entry.is_valid(self.ttl_days):
                logger.info(f"Cache entry {key} is invalid, removing")
                self._remove_entry(key)
                return None
                
            # Update access time
            entry.timestamp = time.time()
            self._save_metadata()
            
            logger.info(f"Cache hit: {key}")
            return entry.path
    
    def cache_model(self, key: str, model_path: str, params: Dict[str, Any]) -> str:
        """Cache a converted model"""
        with self._lock:
            # Check cache size before adding
            self._enforce_size_limit()
            
            # Copy model to cache
            cache_path = self.models_dir / f"{key}.cache"
            shutil.copy2(model_path, cache_path)
            
            # Create metadata entry
            entry = CacheEntry(
                key=key,
                path=str(cache_path),
                timestamp=time.time(),
                size_bytes=cache_path.stat().st_size,
                model_hash=compute_file_hash(str(cache_path)),
                params=params
            )
            
            self.metadata[key] = entry
            self._save_metadata()
            
            logger.info(f"Cached model: {key} ({entry.size_bytes / 1024 / 1024:.1f} MB)")
            return str(cache_path)
    
    def cache_calibration_data(self, name: str, data_path: str) -> str:
        """Cache calibration dataset"""
        cache_path = self.calibration_dir / Path(data_path).name
        
        if not cache_path.exists():
            shutil.copy2(data_path, cache_path)
            logger.info(f"Cached calibration data: {name}")
            
        return str(cache_path)
    
    def get_cached_calibration(self, name: str) -> Optional[str]:
        """Get cached calibration data path"""
        expected_files = {
            'default': 'calibration_images.tgz',
            'fire': 'fire_calibration.tgz',
            'coco_val': 'val2017.zip',
            'diverse': 'diverse_calibration.tgz'
        }
        
        if name in expected_files:
            cache_path = self.calibration_dir / expected_files[name]
            if cache_path.exists():
                return str(cache_path)
                
        return None
    
    def _remove_entry(self, key: str):
        """Remove a cache entry"""
        if key in self.metadata:
            entry = self.metadata[key]
            
            # Remove file
            try:
                Path(entry.path).unlink()
            except FileNotFoundError:
                pass
                
            # Remove metadata
            del self.metadata[key]
    
    def _enforce_size_limit(self):
        """Remove old entries if cache size exceeds limit"""
        total_size = sum(entry.size_bytes for entry in self.metadata.values())
        
        if total_size > self.max_size_bytes:
            # Sort by timestamp (oldest first)
            sorted_entries = sorted(
                self.metadata.items(),
                key=lambda x: x[1].timestamp
            )
            
            # Remove oldest entries until under limit
            for key, entry in sorted_entries:
                if total_size <= self.max_size_bytes:
                    break
                    
                logger.info(f"Removing old cache entry: {key}")
                self._remove_entry(key)
                total_size -= entry.size_bytes
    
    def clear_cache(self):
        """Clear all cached models"""
        with self._lock:
            # Remove all model files
            for entry in self.metadata.values():
                try:
                    Path(entry.path).unlink()
                except FileNotFoundError:
                    pass
                    
            # Clear metadata
            self.metadata.clear()
            self._save_metadata()
            
            logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self.metadata.values())
            valid_entries = sum(1 for entry in self.metadata.values() if entry.is_valid(self.ttl_days))
            
            return {
                'total_entries': len(self.metadata),
                'valid_entries': valid_entries,
                'total_size_mb': total_size / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'usage_percent': (total_size / self.max_size_bytes) * 100,
                'oldest_entry': min((e.timestamp for e in self.metadata.values()), default=0),
                'newest_entry': max((e.timestamp for e in self.metadata.values()), default=0)
            }
    
    def list_cached_models(self) -> List[Tuple[str, Dict[str, Any]]]:
        """List all cached models with their metadata"""
        with self._lock:
            models = []
            for key, entry in self.metadata.items():
                info = {
                    'path': entry.path,
                    'size_mb': entry.size_bytes / 1024 / 1024,
                    'age_days': (time.time() - entry.timestamp) / (24 * 3600),
                    'valid': entry.is_valid(self.ttl_days),
                    'params': entry.params
                }
                models.append((key, info))
            return models


# Singleton instance for global cache
_global_cache = None
_cache_lock = threading.Lock()


def get_global_cache(cache_dir: str = "cache") -> ModelCacheManager:
    """Get or create global cache instance"""
    global _global_cache
    
    with _cache_lock:
        if _global_cache is None:
            _global_cache = ModelCacheManager(cache_dir)
        return _global_cache


if __name__ == "__main__":
    # Example usage
    cache = ModelCacheManager()
    
    # Show cache stats
    stats = cache.get_cache_stats()
    print("Cache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # List cached models
    print("\nCached Models:")
    for key, info in cache.list_cached_models():
        print(f"  {key}:")
        print(f"    Size: {info['size_mb']:.1f} MB")
        print(f"    Age: {info['age_days']:.1f} days")
        print(f"    Valid: {info['valid']}")