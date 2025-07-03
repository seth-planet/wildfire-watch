#!/usr/bin/env python3
"""Prepare calibration dataset for Hailo quantization.

This script extracts and prepares the wildfire calibration dataset for use
with Hailo's INT8 quantization process. It handles downloading, extraction,
and organization of calibration images.

The calibration dataset is critical for maintaining accuracy during INT8
quantization. For wildfire detection, we use a diverse set of images that
include both fire and non-fire scenarios to ensure balanced quantization.

Usage:
    python3 prepare_calibration_data.py --output ./calibration_data/
    python3 prepare_calibration_data.py --input wildfire_calibration_data.tar.gz --output ./calib/

Requirements:
    - Python 3.x (works with any Python 3 version)
    - wget or curl for downloading
    - tar for extraction
"""

import os
import sys
import argparse
import logging
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
import urllib.request
import urllib.error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Calibration dataset URL
CALIBRATION_URL = "https://huggingface.co/datasets/mailseth/wildfire-watch/resolve/main/wildfire_calibration_data.tar.gz?download=true"
CALIBRATION_FILENAME = "wildfire_calibration_data.tar.gz"


class CalibrationDataPreparer:
    """Prepares calibration dataset for model quantization.
    
    This class handles downloading, extracting, and organizing calibration
    images for use with various quantization tools (Hailo, TensorRT, etc.).
    
    Attributes:
        input_path: Path to calibration archive (downloaded if not provided)
        output_path: Directory where calibration images will be extracted
        max_images: Maximum number of images to use (None for all)
        resize: Target size for images (width, height) or None
    """
    
    def __init__(
        self,
        output_path: str,
        input_path: Optional[str] = None,
        max_images: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None
    ):
        """Initialize calibration data preparer.
        
        Args:
            output_path: Directory for extracted calibration images
            input_path: Path to calibration archive (downloaded if not provided)
            max_images: Limit number of images (None for all)
            resize: Resize images to (width, height) if specified
        """
        self.output_path = Path(output_path)
        self.input_path = Path(input_path) if input_path else None
        self.max_images = max_images
        self.resize = resize
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def download_calibration_data(self) -> Path:
        """Download calibration dataset from HuggingFace.
        
        Returns:
            Path to downloaded archive
            
        Raises:
            RuntimeError: If download fails
        """
        # Check if already exists in current directory
        local_path = Path(CALIBRATION_FILENAME)
        if local_path.exists():
            logger.info(f"Using existing calibration archive: {local_path}")
            return local_path
        
        # Download to temporary location
        logger.info(f"Downloading calibration data from HuggingFace...")
        logger.info(f"URL: {CALIBRATION_URL}")
        
        try:
            # Show progress during download
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded / total_size * 100, 100)
                progress_mb = downloaded / 1024 / 1024
                total_mb = total_size / 1024 / 1024
                print(f"\rDownloading: {percent:.1f}% ({progress_mb:.1f}/{total_mb:.1f} MB)", 
                      end='', flush=True)
            
            urllib.request.urlretrieve(CALIBRATION_URL, local_path, reporthook=download_progress)
            print()  # New line after progress
            
            logger.info(f"Downloaded to: {local_path} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return local_path
            
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to download calibration data: {e}")
        except Exception as e:
            # Clean up partial download
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"Download error: {e}")
    
    def extract_archive(self, archive_path: Path) -> List[Path]:
        """Extract calibration images from archive.
        
        Args:
            archive_path: Path to tar.gz archive
            
        Returns:
            List of extracted image paths
            
        Raises:
            RuntimeError: If extraction fails
        """
        logger.info(f"Extracting calibration images to: {self.output_path}")
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        
        if not tarfile.is_tarfile(archive_path):
            raise ValueError(f"Not a valid tar archive: {archive_path}")
        
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                # Get list of image files in archive
                members = tar.getmembers()
                image_members = [
                    m for m in members 
                    if m.isfile() and Path(m.name).suffix.lower() in image_extensions
                ]
                
                logger.info(f"Found {len(image_members)} images in archive")
                
                # Apply max_images limit if specified
                if self.max_images and len(image_members) > self.max_images:
                    logger.info(f"Limiting to {self.max_images} images")
                    image_members = image_members[:self.max_images]
                
                # Extract images
                for i, member in enumerate(image_members):
                    if i % 100 == 0:
                        print(f"\rExtracting: {i}/{len(image_members)} images", end='', flush=True)
                    
                    # Extract to output directory with flattened structure
                    member.name = Path(member.name).name  # Just filename
                    tar.extract(member, self.output_path)
                    
                    extracted_path = self.output_path / member.name
                    image_files.append(extracted_path)
                
                print(f"\rExtracted: {len(image_members)} images")
                
        except tarfile.TarError as e:
            raise RuntimeError(f"Failed to extract archive: {e}")
        except Exception as e:
            raise RuntimeError(f"Extraction error: {e}")
        
        return image_files
    
    def resize_images(self, image_files: List[Path]) -> None:
        """Resize images to target dimensions.
        
        Args:
            image_files: List of image paths to resize
        """
        if not self.resize:
            return
        
        logger.info(f"Resizing images to {self.resize[0]}x{self.resize[1]}...")
        
        # Check if OpenCV is available
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available, skipping resize")
            logger.warning("Install with: pip install opencv-python")
            return
        
        for i, img_path in enumerate(image_files):
            if i % 50 == 0:
                print(f"\rResizing: {i}/{len(image_files)} images", end='', flush=True)
            
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Failed to read image: {img_path}")
                    continue
                
                # Resize
                resized = cv2.resize(img, self.resize, interpolation=cv2.INTER_LINEAR)
                
                # Save back
                cv2.imwrite(str(img_path), resized)
                
            except Exception as e:
                logger.warning(f"Failed to resize {img_path}: {e}")
        
        print(f"\rResized: {len(image_files)} images")
    
    def create_metadata(self, image_files: List[Path]) -> Path:
        """Create metadata file for calibration dataset.
        
        Args:
            image_files: List of calibration images
            
        Returns:
            Path to metadata file
        """
        metadata = {
            'dataset': 'wildfire_calibration',
            'version': '1.0',
            'num_images': len(image_files),
            'image_format': 'mixed',
            'description': 'Calibration dataset for wildfire detection models',
            'categories': ['fire', 'smoke', 'background'],
            'preprocessing': {
                'normalization': {
                    'mean': [0.0, 0.0, 0.0],
                    'std': [255.0, 255.0, 255.0]
                },
                'resize': self.resize if self.resize else 'original'
            }
        }
        
        # Save metadata
        import json
        metadata_path = self.output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created metadata: {metadata_path}")
        return metadata_path
    
    def prepare(self) -> Path:
        """Run complete calibration data preparation.
        
        Returns:
            Path to prepared calibration directory
            
        Raises:
            RuntimeError: If preparation fails
        """
        try:
            # Step 1: Get calibration archive
            if self.input_path and self.input_path.exists():
                archive_path = self.input_path
                logger.info(f"Using provided archive: {archive_path}")
            else:
                archive_path = self.download_calibration_data()
            
            # Step 2: Extract images
            image_files = self.extract_archive(archive_path)
            
            if not image_files:
                raise RuntimeError("No images found in calibration archive")
            
            # Step 3: Resize if requested
            if self.resize:
                self.resize_images(image_files)
            
            # Step 4: Create metadata
            self.create_metadata(image_files)
            
            # Summary
            logger.info(f"\nCalibration data prepared successfully!")
            logger.info(f"Output directory: {self.output_path}")
            logger.info(f"Number of images: {len(image_files)}")
            
            # Show sample files
            sample_files = sorted(image_files)[:5]
            logger.info("Sample files:")
            for f in sample_files:
                logger.info(f"  - {f.name}")
            
            return self.output_path
            
        except Exception as e:
            logger.error(f"Calibration preparation failed: {e}")
            raise


def verify_calibration_data(calib_dir: Path) -> bool:
    """Verify calibration dataset is valid.
    
    Args:
        calib_dir: Path to calibration directory
        
    Returns:
        True if valid, False otherwise
    """
    if not calib_dir.exists():
        logger.error(f"Calibration directory not found: {calib_dir}")
        return False
    
    # Check for images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    
    for ext in image_extensions:
        images.extend(calib_dir.glob(f'*{ext}'))
        images.extend(calib_dir.glob(f'*{ext.upper()}'))
    
    if not images:
        logger.error("No images found in calibration directory")
        return False
    
    logger.info(f"Calibration dataset verified: {len(images)} images")
    
    # Check metadata
    metadata_path = calib_dir / 'metadata.json'
    if metadata_path.exists():
        logger.info("Metadata file found")
    else:
        logger.warning("No metadata file found (not critical)")
    
    return True


def main():
    """Command-line interface for calibration data preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare calibration dataset for model quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for calibration images'
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Path to calibration archive (downloaded if not provided)'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        help='Maximum number of images to extract'
    )
    
    parser.add_argument(
        '--resize',
        help='Resize images to WIDTHxHEIGHT (e.g., 640x640)'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing calibration data'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse resize if provided
    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split('x'))
            resize = (width, height)
        except ValueError:
            logger.error(f"Invalid resize format: {args.resize}")
            logger.error("Use format: WIDTHxHEIGHT (e.g., 640x640)")
            sys.exit(1)
    
    # Verify only mode
    if args.verify_only:
        calib_dir = Path(args.output)
        if verify_calibration_data(calib_dir):
            print(f"\n✓ Calibration data is valid: {calib_dir}")
            sys.exit(0)
        else:
            print(f"\n✗ Calibration data is invalid or missing: {calib_dir}")
            sys.exit(1)
    
    # Prepare calibration data
    try:
        preparer = CalibrationDataPreparer(
            output_path=args.output,
            input_path=args.input,
            max_images=args.max_images,
            resize=resize
        )
        
        calib_dir = preparer.prepare()
        
        # Verify the prepared data
        if verify_calibration_data(calib_dir):
            print(f"\n✓ Calibration data ready: {calib_dir}")
            sys.exit(0)
        else:
            print(f"\n✗ Calibration data preparation failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Preparation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()