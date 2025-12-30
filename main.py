import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from collections import deque

# ===============================
# CONFIG
# ===============================
VIDEO_PATH = "./assets/nohelmet.mp4"
OUTPUT_DIR = "violations"
OUTPUT_VIDEO = "output_traffic.avi"
CONF_THRESHOLD = 0.45

# Define traffic lights with their controlled lanes
# Format: (y1, y2, x1, x2, lane_polygon, lane_name, stop_line_y, direction)
# direction: the direction vehicles are moving in this lane (used to determine which light they face)

# North-South lane (vehicles moving upward/downward)
LANE_NS = np.array([(450, 720), (900, 720), (760, 360), (560, 360)], np.int32)
STOP_LINE_NS = 300

# East-West lane (vehicles moving left/right)
LANE_EW = np.array([(100, 720), (400, 720), (350, 360), (150, 360)], np.int32)
STOP_LINE_EW = 400

TRAFFIC_LIGHTS = [
    {
        "roi": (50, 180, 1050, 1180),  # Traffic light ROI (y1, y2, x1, x2)
        "lane_polygon": LANE_NS,
        "stop_line": STOP_LINE_NS,
        "lane_name": "North-South",
        "direction": "vertical",  # vehicles moving up/down
        "light_position": "top"  # light is at top, controls vehicles at bottom
    },
    {
        "roi": (50, 180, 100, 230),
        "lane_polygon": LANE_EW,
        "stop_line": STOP_LINE_EW,
        "lane_name": "East-West",
        "direction": "horizontal",  # vehicles moving left/right
        "light_position": "left"  # light is at left, controls vehicles on right
    }
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# MODELS
# ===============================
model = YOLO("yolov8n.pt")

# ===============================
# VEHICLE TRACKING
# ===============================
vehicle_tracks = {}  # track_id: deque of positions
next_track_id = 0

def get_vehicle_direction(positions):
    """Calculate vehicle movement direction from position history"""
    if len(positions) < 2:
        return None
    
    dx = positions[-1][0] - positions[0][0]
    dy = positions[-1][1] - positions[0][1]
    
    # Determine primary direction
    if abs(dx) > abs(dy):
        return "horizontal", "right" if dx > 0 else "left"
    else:
        return "vertical", "down" if dy > 0 else "up"

def match_vehicle_to_track(bbox, existing_tracks, max_distance=50):
    """Match detected vehicle to existing track"""
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    
    best_match = None
    best_dist = max_distance
    
    for track_id, positions in existing_tracks.items():
        if len(positions) > 0:
            last_pos = positions[-1]
            dist = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_match = track_id
    
    return best_match

# ===============================
# HELPERS
# ===============================
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

def inside_lane(bbox, lane_poly):
    cx = (bbox[0]+bbox[2])//2
    cy = bbox[3]  # bottom of vehicle
    return cv2.pointPolygonTest(lane_poly, (float(cx), float(cy)), False) >= 0

def is_red_light(frame, roi):
    y1, y2, x1, x2 = roi
    if y2 > frame.shape[0] or x2 > frame.shape[1]:
        return False
    roi_img = frame[y1:y2, x1:x2]
    if roi_img.size == 0:
        return False
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    # Red color ranges in HSV
    mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    total_pixels = roi_img.shape[0] * roi_img.shape[1]
    red_ratio = cv2.countNonZero(mask1 + mask2) / total_pixels
    return red_ratio > 0.12

def vehicle_approaching_light(bbox, track_id, light_config):
    """Check if vehicle is approaching the traffic light it needs to obey"""
    if track_id not in vehicle_tracks or len(vehicle_tracks[track_id]) < 3:
        return False
    
    # Get vehicle direction
    direction_info = get_vehicle_direction(vehicle_tracks[track_id])
    if direction_info is None:
        return False
    
    axis, movement = direction_info
    
    # Check if vehicle's direction matches the lane direction
    if axis != light_config["direction"]:
        return False
    
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    light_roi = light_config["roi"]
    light_cx = (light_roi[2] + light_roi[3]) // 2
    light_cy = (light_roi[0] + light_roi[1]) // 2
    
    # Check if vehicle is moving TOWARDS the light
    if axis == "vertical":
        if light_config["light_position"] == "top":
            # Light is above, vehicle should be moving up
            return movement == "up" and cy > light_cy
        else:
            # Light is below, vehicle should be moving down
            return movement == "down" and cy < light_cy
    else:  # horizontal
        if light_config["light_position"] == "left":
            # Light is on left, vehicle should be moving left
            return movement == "left" and cx > light_cx
        else:
            # Light is on right, vehicle should be moving right
            return movement == "right" and cx < light_cx

def crossed_stop_line(bbox, light_config):
    """Check if vehicle has crossed the stop line"""
    cy = bbox[3]  # bottom of vehicle
    stop_line = light_config["stop_line"]
    
    if light_config["direction"] == "vertical":
        if light_config["light_position"] == "top":
            return cy < stop_line  # crossed upward
        else:
            return cy > stop_line  # crossed downward
    else:  # horizontal
        cx = (bbox[0] + bbox[2]) // 2
        if light_config["light_position"] == "left":
            return cx < stop_line  # crossed leftward
        else:
            return cx > stop_line  # crossed rightward

def save_violation(frame, label, bbox, plate_bbox=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"{OUTPUT_DIR}/{label}_{ts}.jpg"

    out = frame.copy()
    x1, y1, x2, y2 = bbox

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(out, label, (x1, max(y1-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(frame, label, (x1, max(y1-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if plate_bbox:
        px1, py1, px2, py2 = plate_bbox
        cv2.rectangle(out, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(out, "PLATE", (px1, py1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(frame, "PLATE", (px1, py1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite(path, out)
    print(f"[VIOLATION] {label} saved -> {path}")

# ===============================
# VIDEO SETUP
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Could not open video at {VIDEO_PATH}")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 25

print(f"[INFO] Video: {W}x{H} @ {FPS} FPS")
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))
if not writer.isOpened():
    raise RuntimeError("VideoWriter failed to open. Codec issue.")

# ===============================
# TRACKING STATE
# ===============================
captured = set()
frame_count = 0

# ===============================
# MAIN LOOP
# ===============================
print("[INFO] Processing video...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"[INFO] Processing frame {frame_count}...")

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        writer.write(frame)
        continue

    boxes = results[0].boxes
    persons, bikes, helmets, vehicles = [], [], [], []
    current_vehicle_ids = set()

    for b in boxes:
        label = model.names[int(b.cls[0])]
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        bbox = (x1, y1, x2, y2)

        if label == "person":
            persons.append(bbox)
        elif label == "motorcycle":
            bikes.append(bbox)
            vehicles.append(("bike", bbox))
        elif label in ["helmet", "hat"]:
            helmets.append(bbox)
        elif label in ["car", "bus", "truck"]:
            vehicles.append((label, bbox))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ===============================
    # TRACK VEHICLES
    # ===============================
    for lbl, bbox in vehicles:
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        
        track_id = match_vehicle_to_track(bbox, vehicle_tracks)
        
        if track_id is None:
            track_id = next_track_id
            next_track_id += 1
            vehicle_tracks[track_id] = deque(maxlen=10)
        
        vehicle_tracks[track_id].append((cx, cy))
        current_vehicle_ids.add(track_id)

    # Remove old tracks
    vehicle_tracks = {k: v for k, v in vehicle_tracks.items() if k in current_vehicle_ids}

    # ===============================
    # HELMET VIOLATION
    # ===============================
    for bike in bikes:
        bx1, by1, bx2, by2 = bike
        riders = [p for p in persons if iou(p, bike) > 0.35 and p[3] < by2]
        for rider in riders:
            head = (rider[0], rider[1], rider[2], rider[1] + (rider[3]-rider[1])//3)
            has_helmet = any(iou(head, h) > 0.2 for h in helmets)
            vid = f"helmet_{bx1//40}_{by1//40}"
            if not has_helmet and vid not in captured:
                save_violation(frame, "NO_HELMET", rider)
                captured.add(vid)

    # ===============================
    # RED LIGHT VIOLATIONS (DIRECTIONAL)
    # ===============================
    for light_config in TRAFFIC_LIGHTS:
        red = is_red_light(frame, light_config["roi"])
        
        if not red:
            continue
        
        for idx, (lbl, bbox) in enumerate(vehicles):
            # Check if vehicle is in the controlled lane
            if not inside_lane(bbox, light_config["lane_polygon"]):
                continue
            
            # Get track ID for this vehicle
            track_id = match_vehicle_to_track(bbox, vehicle_tracks)
            if track_id is None:
                continue
            
            # Check if vehicle is approaching the light (not moving away)
            if not vehicle_approaching_light(bbox, track_id, light_config):
                continue
            
            # Check if vehicle has crossed the stop line
            if not crossed_stop_line(bbox, light_config):
                continue
            
            vid = f"red_{lbl}_{track_id}_{light_config['lane_name']}"
            if vid not in captured:
                save_violation(frame, f"RED_LIGHT_{light_config['lane_name']}", bbox)
                captured.add(vid)

    # ===============================
    # OVERLAYS
    # ===============================
    for light_config in TRAFFIC_LIGHTS:
        y1, y2, x1, x2 = light_config["roi"]
        lane_poly = light_config["lane_polygon"]
        
        # Draw traffic light ROI
        red = is_red_light(frame, light_config["roi"])
        color = (0, 0, 255) if red else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw lane polygon
        cv2.polylines(frame, [lane_poly], True, (255, 0, 0), 2)
        
        # Draw stop line
        stop_line = light_config["stop_line"]
        if light_config["direction"] == "vertical":
            cv2.line(frame, (0, stop_line), (W, stop_line), (0, 255, 255), 3)
        else:
            cv2.line(frame, (stop_line, 0), (stop_line, H), (0, 255, 255), 3)
        
        # Label
        status = "RED" if red else "GREEN"
        cv2.putText(frame, f"{light_config['lane_name']}: {status}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    writer.write(frame)

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"\n[SUCCESS] Processed {frame_count} frames")
print(f"[SUCCESS] Output video saved as: {OUTPUT_VIDEO}")
print(f"[SUCCESS] {len(captured)} violations detected and saved to {OUTPUT_DIR}/")