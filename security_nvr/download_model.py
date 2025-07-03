#!/usr/bin/env python3
"""Download model files at runtime if they don't exist locally."""

import os
import sys
import urllib.request
from pathlib import Path

def download_model(url, path):
    """Download a model file if it doesn't exist."""
    path = Path(path)
    if path.exists():
        print(f"Model already exists: {path}")
        return True
    
    print(f"Downloading model: {path.name}")
    print(f"  From: {url}")
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
        print(f"  Downloaded successfully: {path.stat().st_size / (1024*1024):.1f} MB")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) == 3:
        download_model(sys.argv[1], sys.argv[2])
    else:
        print("Usage: download_model.py <url> <path>")