import cv2
from collections import Counter
from datetime import datetime
import os

# Prevent repeated logging
ALERT_STATE = {
    "traffic": False,
    "crowd": False
}

LOG_FILE = "output/alerts.log"
os.makedirs("output", exist_ok=True)

def log_alert(alert_type, count, threshold):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(
            f"[{timestamp}] {alert_type} ALERT | count={count} | threshold={threshold}\n"
        )


# ===============================
# Count detected objects
# ===============================
def count_objects(results, model):
    counts = Counter()

    if results.boxes is not None:
        for cls_id in results.boxes.cls.tolist():
            class_name = model.names[int(cls_id)]
            counts[class_name] += 1

    return counts


# ===============================
# Compute dynamic thresholds
# ===============================
def get_dynamic_thresholds(frame):
    h, w = frame.shape[:2]
    area = h * w

    # Thresholds scale with frame size
    if area < 400_000:           # small webcam / image
        traffic_threshold = 5
        crowd_threshold = 6
    elif area < 800_000:         # medium
        traffic_threshold = 8
        crowd_threshold = 9
    else:                        # large / wide view
        traffic_threshold = 12
        crowd_threshold = 14

    return traffic_threshold, crowd_threshold


# ===============================
# Draw alert box (safe for small images)
# ===============================
def draw_alert(frame, text, y_offset=10, bg_color=(0, 0, 255)):
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    padding = 8

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x2 = w - 10
    x1 = max(10, x2 - tw - padding * 2)
    y1 = y_offset
    y2 = y1 + th + padding * 2

    # Background box
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)

    # Text
    cv2.putText(
        frame,
        text,
        (x1 + padding, y2 - padding),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )

    return y2 + 10


# ===============================
# Traffic & Crowd alert logic (DYNAMIC)
# ===============================
def apply_alerts(frame, object_counts):
    traffic_count = object_counts.get("car", 0) + object_counts.get("truck", 0)
    person_count = object_counts.get("person", 0)

    traffic_th, crowd_th = get_dynamic_thresholds(frame)

    y = 10

    # -------- TRAFFIC --------
    if traffic_count >= traffic_th:
        y = draw_alert(
            frame,
            f"TRAFFIC ALERT ({traffic_count})",
            y,
            (0, 0, 255)
        )

        if not ALERT_STATE["traffic"]:
            log_alert("TRAFFIC", traffic_count, traffic_th)
            ALERT_STATE["traffic"] = True
    else:
        ALERT_STATE["traffic"] = False

    # -------- CROWD --------
    if person_count >= crowd_th:
        draw_alert(
            frame,
            f"CROWD ALERT ({person_count})",
            y,
            (0, 165, 255)
        )

        if not ALERT_STATE["crowd"]:
            log_alert("CROWD", person_count, crowd_th)
            ALERT_STATE["crowd"] = True
    else:
        ALERT_STATE["crowd"] = False
