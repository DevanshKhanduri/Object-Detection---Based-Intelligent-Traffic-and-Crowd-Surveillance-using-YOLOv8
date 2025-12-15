from ultralytics import YOLO
import cv2
import time
import os
from utils import count_objects, apply_alerts

# ===============================
# Load model
# ===============================
model = YOLO("model/best.pt")

# ===============================
# Webcam
# ===============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

os.makedirs("output", exist_ok=True)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

FPS = 10

out = cv2.VideoWriter(
    "output/output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (width, height)
)

WINDOW_NAME = "YOLOv8 Live Detection"
prev_time = 0

print("✅ Webcam started | Press 'q' or close window")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
    annotated_frame = results[0].plot(line_width=1, font_size=0.45)

    # Object counting
    object_counts = count_objects(results[0], model)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Object counts
    y_offset = 60
    for obj, count in object_counts.items():
        cv2.putText(
            annotated_frame,
            f"{obj}: {count}",
            (15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )
        y_offset += 22

    # Alerts
    apply_alerts(annotated_frame, object_counts)

    # Show & save
    cv2.imshow(WINDOW_NAME, annotated_frame)
    out.write(annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("🛑 Webcam stopped | Video saved")
