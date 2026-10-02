"""
KITTI Object Detection data loader.

Design decisions (from NHAT_KY_QUYET_DINH.md §3):
- Loader is split-agnostic: receives a list of frame IDs from outside.
- DontCare and non-vehicle classes are dropped when reading labels.
- Provides both depth (Z) and Euclidean distance; use depth for all metrics.
- Uses P2 (and R0_rect) per image for calibration.
- VEHICLE_CLASSES = ['Car', 'Van', 'Truck'] but results must be reported per-class.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VEHICLE_CLASSES = ["Car", "Van", "Truck"]

# KITTI Hard filter thresholds (§4 of KE_HOACH_V4)
HARD_MIN_HEIGHT = 25       # bbox height >= 25 px
HARD_MAX_OCCLUDED = 2      # occluded <= 2
HARD_MAX_TRUNCATED = 0.50  # truncated <= 0.50


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class KITTICalib:
    """Calibration data for a single KITTI image."""
    P2: np.ndarray              # 3x4 projection matrix (left color camera)
    R0_rect: np.ndarray         # 3x3 rectification rotation
    Tr_velo_to_cam: np.ndarray  # 3x4 Velodyne-to-camera transform

    @property
    def fx(self) -> float:
        return float(self.P2[0, 0])

    @property
    def fy(self) -> float:
        return float(self.P2[1, 1])

    @property
    def cx(self) -> float:
        return float(self.P2[0, 2])

    @property
    def cy(self) -> float:
        return float(self.P2[1, 2])


@dataclass
class KITTIObject:
    """A single labeled object in a KITTI frame."""
    obj_class: str
    truncated: float
    occluded: int          # 0=fully visible, 1=partly, 2=largely, 3=unknown
    alpha: float           # observation angle [-pi, pi]
    bbox: np.ndarray       # [x1, y1, x2, y2] in pixels
    dimensions: np.ndarray # [h, w, l] in meters (height, width, length)
    location: np.ndarray   # [x, y, z] in camera coordinates (meters)
    rotation_y: float      # rotation around Y-axis in camera coords

    @property
    def depth(self) -> float:
        """Depth Z (location_z) — used for all metrics."""
        return float(self.location[2])

    @property
    def distance(self) -> float:
        """Euclidean distance from camera — for reference only."""
        return float(np.linalg.norm(self.location))

    @property
    def bbox_height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def bbox_width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    def passes_hard_filter(self) -> bool:
        """Check if object passes KITTI Hard difficulty filter."""
        return (
            self.bbox_height >= HARD_MIN_HEIGHT
            and self.occluded <= HARD_MAX_OCCLUDED
            and self.truncated <= HARD_MAX_TRUNCATED
        )

    def get_difficulty(self) -> str:
        """Return KITTI difficulty level: Easy, Moderate, Hard, or Excluded."""
        h = self.bbox_height
        occ = self.occluded
        trunc = self.truncated

        if h >= 40 and occ == 0 and trunc <= 0.15:
            return "Easy"
        elif h >= 25 and occ <= 1 and trunc <= 0.30:
            return "Moderate"
        elif h >= 25 and occ <= 2 and trunc <= 0.50:
            return "Hard"
        else:
            return "Excluded"


@dataclass
class KITTIFrame:
    """All data for a single KITTI frame."""
    frame_id: str                          # e.g. "000000"
    calib: KITTICalib
    objects: list                           # list of KITTIObject
    image_path: Optional[str] = None
    image_size: Optional[tuple] = None     # (width, height)
    drive: Optional[str] = None            # e.g. "2011_09_26_drive_0005_sync"


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------
def parse_calib(calib_path: str) -> KITTICalib:
    """Parse a KITTI calibration file."""
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, values = line.partition(":")
            key = key.strip()
            vals = np.array([float(x) for x in values.split()])
            data[key] = vals

    P2 = data["P2"].reshape(3, 4)
    R0_rect = data["R0_rect"].reshape(3, 3)
    Tr_velo = data["Tr_velo_to_cam"].reshape(3, 4)

    return KITTICalib(P2=P2, R0_rect=R0_rect, Tr_velo_to_cam=Tr_velo)


def parse_label(label_path: str, vehicle_only: bool = True) -> list:
    """
    Parse a KITTI label file.

    Args:
        label_path: Path to .txt label file.
        vehicle_only: If True, only return objects in VEHICLE_CLASSES
                      (drops DontCare, Pedestrian, Cyclist, Misc, etc.)
    Returns:
        List of KITTIObject.
    """
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            obj_class = parts[0]

            # Skip non-vehicle classes
            if vehicle_only and obj_class not in VEHICLE_CLASSES:
                continue

            obj = KITTIObject(
                obj_class=obj_class,
                truncated=float(parts[1]),
                occluded=int(parts[2]),
                alpha=float(parts[3]),
                bbox=np.array([float(parts[4]), float(parts[5]),
                               float(parts[6]), float(parts[7])]),
                dimensions=np.array([float(parts[8]), float(parts[9]),
                                     float(parts[10])]),
                location=np.array([float(parts[11]), float(parts[12]),
                                   float(parts[13])]),
                rotation_y=float(parts[14]),
            )
            objects.append(obj)

    return objects


def read_drive_mapping(mapping_path: str) -> dict:
    """
    Read train_mapping.txt to get frame_id -> drive mapping.

    Each line: "date drive_name raw_frame_id"
    Line number (0-indexed) corresponds to the KITTI Object frame ID
    after applying train_rand.txt permutation.

    However, train_mapping.txt is already in KITTI Object order
    (line i = info for KITTI Object frame i), so we read directly.

    Returns:
        dict: {frame_id_str: drive_name}  e.g. {"000000": "2011_09_26_drive_0005_sync"}
    """
    mapping = {}
    with open(mapping_path, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 2:
                frame_id = f"{idx:06d}"
                drive = parts[1]  # e.g. "2011_09_26_drive_0005_sync"
                mapping[frame_id] = drive
    return mapping


# ---------------------------------------------------------------------------
# Main Loader
# ---------------------------------------------------------------------------
class KITTILoader:
    """
    KITTI Object Detection dataset loader.

    Split-agnostic: receives frame IDs from outside.
    Does NOT contain split logic.
    """

    def __init__(self, data_root: str):
        """
        Args:
            data_root: Path to data/kitti/ directory containing
                       image_2/, label_2/, calib/, devkit/
        """
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / "image_2"
        self.label_dir = self.data_root / "label_2"
        self.calib_dir = self.data_root / "calib"
        self.mapping_path = self.data_root / "devkit" / "mapping" / "train_mapping.txt"

        # Validate directories exist
        for d in [self.image_dir, self.label_dir, self.calib_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Directory not found: {d}")

        # Load drive mapping
        if self.mapping_path.exists():
            self._drive_mapping = read_drive_mapping(str(self.mapping_path))
        else:
            raise FileNotFoundError(
                f"Drive mapping not found: {self.mapping_path}. "
                "Download devkit from KITTI website."
            )

        # All available frame IDs
        self._all_frame_ids = sorted([
            f.stem for f in self.label_dir.glob("*.txt")
        ])

    @property
    def all_frame_ids(self) -> list:
        """All available frame IDs (sorted)."""
        return list(self._all_frame_ids)

    @property
    def num_frames(self) -> int:
        return len(self._all_frame_ids)

    def get_drive(self, frame_id: str) -> str:
        """Get the drive name for a frame ID."""
        return self._drive_mapping[frame_id]

    def get_all_drives(self) -> dict:
        """
        Get mapping of drive -> list of frame IDs.

        Returns:
            dict: {drive_name: [frame_id, ...]}
        """
        drives = {}
        for fid in self._all_frame_ids:
            drive = self._drive_mapping[fid]
            if drive not in drives:
                drives[drive] = []
            drives[drive].append(fid)
        return drives

    def load_frame(self, frame_id: str, load_image: bool = False) -> KITTIFrame:
        """
        Load all data for a single frame.

        Args:
            frame_id: e.g. "000000"
            load_image: If True, also read image to get its size.
        """
        # Calibration
        calib_path = self.calib_dir / f"{frame_id}.txt"
        calib = parse_calib(str(calib_path))

        # Labels
        label_path = self.label_dir / f"{frame_id}.txt"
        objects = parse_label(str(label_path), vehicle_only=True)

        # Image path & size
        image_path = str(self.image_dir / f"{frame_id}.png")
        image_size = None
        if load_image:
            from PIL import Image
            with Image.open(image_path) as img:
                image_size = img.size  # (width, height)

        # Drive
        drive = self._drive_mapping.get(frame_id)

        return KITTIFrame(
            frame_id=frame_id,
            calib=calib,
            objects=objects,
            image_path=image_path,
            image_size=image_size,
            drive=drive,
        )

    def load_frames(self, frame_ids: list, load_image: bool = False) -> list:
        """Load multiple frames."""
        return [self.load_frame(fid, load_image) for fid in frame_ids]

    def get_statistics(self, frame_ids: list = None) -> dict:
        """
        Compute basic statistics over a set of frames.

        Args:
            frame_ids: List of frame IDs. If None, use all.

        Returns:
            dict with keys: n_frames, n_objects, n_per_class,
                           depth_stats, difficulty_counts
        """
        if frame_ids is None:
            frame_ids = self._all_frame_ids

        stats = {
            "n_frames": len(frame_ids),
            "n_objects": 0,
            "n_per_class": {},
            "depths": [],
            "difficulty_counts": {"Easy": 0, "Moderate": 0, "Hard": 0, "Excluded": 0},
        }

        for fid in frame_ids:
            frame = self.load_frame(fid)
            for obj in frame.objects:
                stats["n_objects"] += 1
                cls = obj.obj_class
                stats["n_per_class"][cls] = stats["n_per_class"].get(cls, 0) + 1
                stats["depths"].append(obj.depth)
                diff = obj.get_difficulty()
                stats["difficulty_counts"][diff] += 1

        depths = np.array(stats["depths"]) if stats["depths"] else np.array([])
        stats["depth_stats"] = {
            "min": float(depths.min()) if len(depths) > 0 else None,
            "max": float(depths.max()) if len(depths) > 0 else None,
            "mean": float(depths.mean()) if len(depths) > 0 else None,
            "median": float(np.median(depths)) if len(depths) > 0 else None,
        }

        # Depth range distribution (§6 of KE_HOACH_V4)
        bins = [0, 10, 20, 30, 50, float("inf")]
        labels = ["0-10m", "10-20m", "20-30m", "30-50m", ">50m"]
        stats["depth_distribution"] = {}
        for i in range(len(bins) - 1):
            mask = (depths >= bins[i]) & (depths < bins[i + 1])
            stats["depth_distribution"][labels[i]] = int(mask.sum())

        return stats
