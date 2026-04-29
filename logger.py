"""
logger.py - Logging Module
Flood Evacuation Route Optimizer
Logs detections, alerts, and system events to log files.
"""
 
import logging
import csv
import json
import os
from datetime import datetime
from config import LOG_FILE_PATH, DETECTION_LOG_PATH, ALERT_LOG_PATH, LOG_DIR
 
# ─────────────────────────────────────────
# SYSTEM LOGGER SETUP
# ─────────────────────────────────────────
 
os.makedirs(LOG_DIR, exist_ok=True)
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()           # Also print to console
    ]
)
 
logger = logging.getLogger("FloodSystem")
 
 
# ─────────────────────────────────────────
# DETECTION LOGGER
# ─────────────────────────────────────────
 
class DetectionLogger:
    """Logs YOLOv8 detections to CSV file."""
 
    def __init__(self, filepath: str = DETECTION_LOG_PATH):
        self.filepath = filepath
        self._init_csv()
 
    def _init_csv(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "camera_id", "class_name",
                    "confidence", "x1", "y1", "x2", "y2",
                    "risk_level"
                ])
 
    def log_detection(self, camera_id: str, class_name: str,
                      confidence: float, bbox: tuple, risk_level: str):
        """Log a single detection to CSV."""
        x1, y1, x2, y2 = bbox
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                camera_id, class_name,
                round(confidence, 4),
                x1, y1, x2, y2,
                risk_level
            ])
 
    def log_zone_status(self, zone_id: str, risk_level: str,
                        people_stranded: int, vehicles_stuck: int,
                        sos_detected: bool):
        """Log overall zone assessment."""
        logger.info(
            f"ZONE [{zone_id}] | Risk: {risk_level} | "
            f"Stranded: {people_stranded} | Vehicles: {vehicles_stuck} | "
            f"SOS: {sos_detected}"
        )
 
 
# ─────────────────────────────────────────
# ALERT LOGGER
# ─────────────────────────────────────────
 
class AlertLogger:
    """Logs evacuation alerts to JSON file."""
 
    def __init__(self, filepath: str = ALERT_LOG_PATH):
        self.filepath = filepath
        self.alerts   = self._load_existing()
 
    def _load_existing(self) -> list:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []
 
    def log_alert(self, zone_id: str, risk_level: str,
                  route_path: list, message: str,
                  distance_km: float, time_min: float):
        """Log an evacuation alert."""
        alert = {
            "timestamp":   datetime.now().isoformat(),
            "zone_id":     zone_id,
            "risk_level":  risk_level,
            "route_path":  route_path,
            "distance_km": distance_km,
            "time_min":    time_min,
            "message":     message,
        }
        self.alerts.append(alert)
        self._save()
 
        if risk_level == "HIGH":
            logger.critical(f"🚨 HIGH RISK ALERT — Zone: {zone_id} | Route: {' → '.join(route_path)}")
        else:
            logger.warning(f"⚠️  ALERT — Zone: {zone_id} | Risk: {risk_level}")
 
    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.alerts, f, indent=2)
 
    def get_recent(self, n: int = 10) -> list:
        return self.alerts[-n:]
 
 
# ─────────────────────────────────────────
# ROUTE LOGGER
# ─────────────────────────────────────────
 
def log_route_update(route_path: list, distance: float, time_min: float, warnings: list):
    """Log route recalculation event."""
    logger.info(
        f"ROUTE UPDATE | Path: {' → '.join(route_path)} | "
        f"Distance: {distance}km | ETA: {time_min}min | "
        f"Warnings: {len(warnings)}"
    )
 
def log_system_start():
    logger.info("=" * 55)
    logger.info("🌊 Flood Evacuation System STARTED")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 55)
 
def log_system_stop():
    logger.info("🛑 Flood Evacuation System STOPPED")
    logger.info("=" * 55)
 
def log_model_loaded(model_name: str):
    logger.info(f"✅ Model loaded: {model_name}")
 
def log_inference_stats(fps: float, inference_ms: float, frame_count: int):
    logger.info(f"📊 FPS: {fps:.1f} | Inference: {inference_ms:.1f}ms | Frames: {frame_count}")

