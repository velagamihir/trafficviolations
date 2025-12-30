import cv2
import numpy as np
import os
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# ===============================
# CONFIG
# ===============================
VIDEO_PATH = "./assets/samplevideo.mp4"
OUTPUT_DIR = "violations"
OUTPUT_VIDEO = "output_traffix.mp4"
CONF_THRESHOLD = 0.45

RED_LIGHT_LINE_Y = 300

TL_Y1, TL_Y2 = 50, 180
TL_X1, TL_X2 = 1050, 1180

LANE_POLYGON = np.array([
    (450, 720),
    (900, 720),
    (760, 360),
    (560, 360)
], np.int32)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# MODELS
# ===============================
model = YOLO("yolov8n.pt")

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

def inside_lane(bbox):
    cx = (bbox[0]+bbox[2])//2
    cy = bbox[3]
    return cv2.pointPolygonTest(LANE_POLYGON, (float(cx), float(cy)), False) >= 0

def is_red_light(frame):
    # Check bounds before slicing
    if TL_Y2 > frame.shape[0] or TL_X2 > frame.shape[1]:
        return False
    
    roi = frame[TL_Y1:TL_Y2, TL_X1:TL_X2]
    if roi.size == 0:
        return False
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    total_pixels = roi.shape[0] * roi.shape[1]
    return cv2.countNonZero(mask1 + mask2) / total_pixels > 0.12

def save_violation(frame, label, bbox, plate_bbox=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"{OUTPUT_DIR}/{label}_{ts}.jpg"

    out = frame.copy()

    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(out, label, (x1, max(y1-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if plate_bbox:
        px1, py1, px2, py2 = plate_bbox
        cv2.rectangle(out, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(out, "PLATE DETECTED", (px1, py1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(out, "PLATE NOT DETECTED",
                    (x1, y2+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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

FPS = cap.get(cv2.CAP_PROP_FPS)
if FPS == 0 or FPS is None:
    FPS = 25  # fallback

print(f"[INFO] Video: {W}x{H} @ {FPS} FPS")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))

if not writer.isOpened():
    raise RuntimeError("VideoWriter failed to open. Codec issue.")

# ===============================
# TRACKING STATE
# ===============================
captured = set()  # Track violation IDs to prevent duplicates
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

    # Run YOLO detection
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    
    # Check if results exist and have boxes
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        writer.write(frame)
        continue
    
    boxes = results[0].boxes

    persons, bikes, helmets, vehicles = [], [], [], []

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
    # HELMET VIOLATION (STRICT)
    # ===============================
    for bike in bikes:
        bx1, by1, bx2, by2 = bike

        riders = [
            p for p in persons
            if iou(p, bike) > 0.35 and p[3] < by2
        ]

        for rider in riders:
            head = (rider[0], rider[1],
                    rider[2], rider[1] + (rider[3]-rider[1])//3)

            has_helmet = any(iou(head, h) > 0.2 for h in helmets)

            vid = f"helmet_{bx1//40}_{by1//40}"

            if not has_helmet and vid not in captured:
                save_violation(frame, "NO_HELMET", rider)
                captured.add(vid)

    # ===============================
    # RED LIGHT
    # ===============================
    red = is_red_light(frame)

    for lbl, bbox in vehicles:
        vid = f"red_{lbl}_{bbox[0]//40}_{bbox[1]//40}"

        if red and inside_lane(bbox) and bbox[3] > RED_LIGHT_LINE_Y:
            if vid not in captured:
                save_violation(frame, "RED_LIGHT_JUMP", bbox)
                captured.add(vid)

    # ===============================
    # OVERLAYS
    # ===============================
    cv2.line(frame, (0, RED_LIGHT_LINE_Y), (W, RED_LIGHT_LINE_Y), (0, 0, 255), 3)
    cv2.polylines(frame, [LANE_POLYGON], True, (255, 0, 0), 2)
    cv2.putText(frame,
                f"Signal: {'RED' if red else 'GREEN'}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if red else (0, 255, 0), 3)

    writer.write(frame)

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"\n[SUCCESS] Processed {frame_count} frames")
print(f"[SUCCESS] Output video saved as: {OUTPUT_VIDEO}")
print(f"[SUCCESS] {len(captured)} violations detected and saved to {OUTPUT_DIR}/")