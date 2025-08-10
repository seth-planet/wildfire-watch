#!/usr/bin/env python3.10
"""Download test videos for Hailo testing.

This script downloads wildfire detection demo videos from GitHub
and caches them locally for testing purposes.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from hailo_test_utils import VideoDownloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Download test videos for Hailo testing."""
    parser = argparse.ArgumentParser(description="Download wildfire test videos")
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path(__file__).parent / 'test_videos',
        help='Directory to cache videos (default: ./test_videos)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if videos exist'
    )
    parser.add_argument(
        '--video',
        choices=['fire1.mov', 'fire2.mov', 'fire3.mp4', 'fire4.mp4'],
        help='Download specific video only'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = VideoDownloader(cache_dir=args.cache_dir)
    
    print(f"\n=== Wildfire Test Video Downloader ===")
    print(f"Cache directory: {args.cache_dir}")
    print()
    
    try:
        if args.video:
            # Download specific video
            path = downloader.download_video(args.video, force=args.force)
            print(f"✓ Downloaded: {args.video} -> {path}")
        else:
            # Download all videos
            videos = downloader.download_all_videos()
            
            print(f"\nDownloaded {len(videos)} videos:")
            for name, path in videos.items():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {name}: {size_mb:.1f} MB")
                
        print(f"\nVideos cached in: {args.cache_dir}")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()