"""
preprocessing.py - Input Data Preparation & Feature Extraction
Flood Evacuation Route Optimizer
"""
 
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional
from config import (
    FRAME_WIDTH, FRAME_HEIGHT, INPUT_SOURCE,
    CNN_IMAGE_SIZE
)
from logger import logger
 
 
# ─────────────────────────────────────────
# INPUT DATA PREPARATION
# ─────────────────────────────────────────
 
class InputDataPreparation:
    """
    Handles all input sources: webcam, video file.
    Preprocesses frames for YOLOv8 and CNN inference.
    """
 
    def __init__(self, source=INPUT_SOURCE):
        self.source = source
        self.cap    = None
 
    def initialize(self) -> bool:
        """Open video source. Returns True if successful."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"❌ Cannot open input source: {self.source}")
            return False
        logger.info(f"✅ Input source opened: {self.source}")
        return True
 
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from source."""
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame
 
    def preprocess_for_yolov8(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare frame for YOLOv8 inference.
        Steps:
          1. Resize to standard resolution
          2. Enhance contrast for murky flood footage
          3. Convert BGR → RGB
        """
        # Step 1: Resize
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
 
        # Step 2: CLAHE contrast enhancement
        #frame = self._apply_clahe(frame)
 
        # Step 3: Keep as BGR (OpenCV/YOLOv8 expects BGR)
        return frame
 
    def preprocess_for_cnn(self, frame: np.ndarray) -> torch.Tensor:
        """
        Prepare frame for CNN risk classification.
        Returns normalized tensor of shape (1, 3, 224, 224).
        """
        transform = transforms.Compose([
            transforms.Resize((CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            )
        ])
 
        # BGR → RGB → PIL
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        return transform(pil).unsqueeze(0)  # Add batch dimension
 
    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
            logger.info("📷 Input source released.")
 
    # ── Private Helpers ────────────────────
 
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE for better visibility in low-light/murky flood footage."""
        lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b  = cv2.split(lab)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l        = clahe.apply(l)
        lab      = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
 
 
# ─────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────
 
class FeatureExtractor:
    """
    Extracts meaningful features from raw detections
    for downstream routing and alerting decisions.
    """
 
    def extract_flood_features(self, frame: np.ndarray) -> dict:
        """
        Extract visual flood features from a frame.
        Used as supplementary input to CNN classifier.
 
        Features extracted:
          - blue_ratio:    % of blue pixels (water indicator)
          - dark_ratio:    % of dark pixels (deep water)
          - edge_density:  edge density (debris indicator)
          - blur_score:    blurriness (disturbed water)
        """
        hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w       = frame.shape[:2]
        total_px   = h * w
 
        # Blue/water pixel ratio (Hue 90–130 in HSV)
        blue_mask  = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
        blue_ratio = np.count_nonzero(blue_mask) / total_px
 
        # Dark pixel ratio (deep/dirty water)
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_mask  = gray < 50
        dark_ratio = np.count_nonzero(dark_mask) / total_px
 
        # Edge density (road debris, submerged objects)
        edges      = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / total_px
 
        # Blur score (camera shake, disturbed water)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
 
        return {
            "blue_ratio":    round(float(blue_ratio),   4),
            "dark_ratio":    round(float(dark_ratio),   4),
            "edge_density":  round(float(edge_density), 4),
            "blur_score":    round(float(blur_score),   4),
        }
 
    def extract_motion_features(self, prev_frame: np.ndarray,
                                  curr_frame: np.ndarray) -> dict:
        """
        Detect motion between frames — useful for
        detecting rising water and moving crowds.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
 
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
 
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
 
        return {
            "mean_motion":  round(float(np.mean(magnitude)),  4),
            "max_motion":   round(float(np.max(magnitude)),   4),
            "motion_direction": round(float(np.mean(angle)),  4),
        }
 
    def extract_water_level_feature(self, frame: np.ndarray,
                                     roi: tuple = None) -> float:
        """
        Estimate relative water level from a region of interest (ROI).
        Used when physical water level markers are visible in frame.
 
        Args:
            roi: (x1, y1, x2, y2) region containing water marker
        Returns:
            Normalized water level 0.0 (empty) to 1.0 (full)
        """
        if roi:
            x1, y1, x2, y2 = roi
            region = frame[y1:y2, x1:x2]
        else:
            region = frame
 
        hsv       = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
        h         = region.shape[0]
        col_sum   = np.sum(blue_mask, axis=1)
 
        # Find highest row with significant blue presence
        threshold = region.shape[1] * 0.1 * 255
        water_rows = np.where(col_sum > threshold)[0]
 
        if len(water_rows) == 0:
            return 0.0
        highest_water = water_rows.min()
        return round(1.0 - (highest_water / h), 4)
    
    def detect_water_presence(frame):
        """
    Detect water using color threshold (simple heuristic)
    Returns: water_detected (True/False), water_ratio (0-1)
        """
        import cv2
        import numpy as np

    # convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # range for muddy water (brownish)
        lower = np.array([5, 50, 50])
        upper = np.array([25, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        water_pixels = np.sum(mask > 0)
        total_pixels = frame.shape[0] * frame.shape[1]

        water_ratio = water_pixels / total_pixels

        return water_ratio > 0.1, water_ratio
