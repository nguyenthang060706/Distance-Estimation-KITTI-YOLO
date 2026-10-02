"""
scripts/train_detector.py: CLI entry point for training 2D detectors.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.train_detector import train_detector

if __name__ == "__main__":
    train_detector()
