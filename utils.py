"""
utils.py - Utility & Helper Functions
Flood Evacuation Route Optimizer
"""
 
import cv2
import numpy as np
import math
import time
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from config import CLASS_COLORS, CLASS_RISK_WEIGHTS, CNN_RISK_LABELS
 
 
# ─────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────
 
@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    camera_id: str
 
@dataclass
class ZoneStatus:
    zone_id: str
    risk_level: str
    flood_detected: bool
    people_stranded: int
    vehicles_stuck: int
    water_level: str
    sos_detected: bool
    timestamp: float
 
@dataclass
class EvacuationRoute:
    path: List[str]
    total_distance: float
    risk_score: float
    estimated_time_min: float
    road_names: List[str]
    warnings: List[str]
 
 
# ─────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────
 
def draw_detection(frame: np.ndarray, det: Detection) -> np.ndarray:
    """Draw bounding box and label for a detection."""
    x1, y1, x2, y2 = det.bbox
    color = CLASS_COLORS.get(det.class_name, (0, 255, 0))
 
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
 
    label = f"{det.class_name} {det.confidence:.2f}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1-lh-8), (x1+lw+4, y1), color, -1)
    cv2.putText(frame, label, (x1+2, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame
 
def draw_zone_overlay(frame: np.ndarray, zone: ZoneStatus) -> np.ndarray:
    """Draw zone status banner at bottom of frame."""
    risk_colors = {
        "HIGH":   (0, 0, 255),
        "MEDIUM": (0, 165, 255),
        "LOW":    (0, 255, 255),
        "SAFE":   (0, 255, 0),
    }
    color = risk_colors.get(zone.risk_level, (255,255,255))
    h, w = frame.shape[:2]
 
    cv2.rectangle(frame, (0, h-80), (w, h), (0,0,0), -1)
    cv2.putText(frame, f"ZONE: {zone.zone_id}  |  RISK: {zone.risk_level}",
                (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    #cv2.putText(frame,
                #f"Stranded: {zone.people_stranded}  Vehicles: {zone.vehicles_stuck}  SOS: {zone.sos_detected}",
                #(10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
 
    if zone.sos_detected:
        cv2.rectangle(frame, (0, 0), (w, 50), (0,0,255), -1)
        cv2.putText(frame, "SOS DETECTED - DISPATCH RESCUE IMMEDIATELY",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return frame
 
def draw_fps(frame: np.ndarray, fps: float, inference_ms: float) -> np.ndarray:
    """Draw FPS and inference time on frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference: {inference_ms:.1f}ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return frame
 
def draw_route_overlay(frame: np.ndarray, route: EvacuationRoute) -> np.ndarray:
    """Draw evacuation route on top-right corner of frame."""
    if not route:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()
    #cv2.rectangle(overlay, (w-250, 55), (w, 155), (0,0,0), -1)
    cv2.rectangle(overlay, (w-200, 10), (w-10, 90), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
 
    cv2.putText(frame, "EVACUATION ROUTE:", (w-190, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)
    route_str = " -> ".join(route.path[:3])
    if len(route.path) > 4:
        route_str += "..."
    cv2.putText(frame, route_str, (w-190, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    cv2.putText(frame, f"Dist: {route.total_distance}km", (w-190, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    cv2.putText(frame, f"ETA:  {route.estimated_time_min}min", (w-190, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    return frame
 
 
# ─────────────────────────────────────────
# ZONE ASSESSMENT HELPERS
# ─────────────────────────────────────────
 
def assess_zone_risk(detections: List[Detection], zone_id: str) -> ZoneStatus:
    """Calculate overall risk level from list of detections."""

    if not detections:
        return ZoneStatus(zone_id, "SAFE", False, 0, 0, "SAFE", False, time.time())

    counts = {}

    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1

        if det.class_name == "person":
            counts["stranded_person"] = counts.get("stranded_person", 0) + 1

    # Count values
    person_count = counts.get("person", 0) + counts.get("stranded_person", 0)
    flood_count = counts.get("flood", 0)

    # Risk score
    risk_score = sum(
        CLASS_RISK_WEIGHTS.get(d.class_name, 1) * d.confidence
        for d in detections
    )

    from config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD

    # FINAL RISK LOGIC (FIXED)
    if flood_count > 0:
        risk_level = "HIGH"

    elif counts.get("stranded_person", 0) > 0:
        risk_level = "MEDIUM"

    elif risk_score >= HIGH_RISK_THRESHOLD or counts.get("sos_person", 0) > 0:
        risk_level = "HIGH"

    elif risk_score >= MEDIUM_RISK_THRESHOLD:
        risk_level = "MEDIUM"

    elif risk_score >= 1:
        risk_level = "LOW"

    else:
        risk_level = "SAFE"

    # Water level
    if counts.get("water_marker_danger", 0) > 0:
        water_level = "DANGER"
    elif flood_count > 0:
        water_level = "WARNING"
    else:
        water_level = "SAFE"

    return ZoneStatus(
        zone_id=zone_id,
        risk_level=risk_level,
        flood_detected=flood_count > 0,
        people_stranded=counts.get("stranded_person", 0),
        vehicles_stuck = int(flood_count * 1.5),
        water_level=water_level,
        sos_detected=counts.get("sos_person", 0) > 0,
        timestamp=time.time()
    )


CLASS_RISK_WEIGHTS = {
    "flood": 2,
    "person": 1,
    "stranded_person": 2
}


# ─────────────────────────────────────────
# MATH / GEO HELPERS
# ─────────────────────────────────────────
 
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Real-world distance between two GPS coordinates in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
 
def calculate_fps(prev_time: float) -> Tuple[float, float]:
    """Returns (fps, new_time)."""
    curr = time.time()
    fps  = 1.0 / max(curr - prev_time, 1e-6)
    return round(fps, 1), curr
 
 
# ─────────────────────────────────────────
# IMAGE PREPROCESSING HELPERS
# ─────────────────────────────────────────
 
def enhance_flood_image(frame: np.ndarray) -> np.ndarray:
    """
    Enhance image for better flood detection.
    Applies CLAHE contrast enhancement for murky/dark flood footage.
    """
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    lab   = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
 
def resize_frame(frame: np.ndarray, width: int = 1280, height: int = 720) -> np.ndarray:
    return cv2.resize(frame, (width, height))
 
 
# ─────────────────────────────────────────
# VIDEO WRITER HELPER
# ─────────────────────────────────────────
 
def get_video_writer(output_path: str, width: int, height: int, fps: float = 20.0):
    """Create OpenCV VideoWriter for saving annotated output."""
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 
 
# ─────────────────────────────────────────
# ALERT MESSAGE GENERATOR
# ─────────────────────────────────────────
 
def format_alert_message(zone_id: str, risk_level: str,
                          route: EvacuationRoute, node_names: list) -> str:
    """Generate human-readable SMS/notification message."""
    route_str = " → ".join(node_names)
    now = time.strftime("%H:%M:%S")
    return (
        f"FLOOD ALERT [{zone_id}]\n"
        f"Risk: {risk_level}\n"
        f"EVACUATE NOW!\n"
        f"Route: {route_str}\n"
        f"Distance: {route.total_distance}km | ETA: {route.estimated_time_min}min\n"
        f"Updated: {now}"
    )

