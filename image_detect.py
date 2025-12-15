from ultralytics import YOLO
import cv2
from utils import count_objects, apply_alerts

# ===============================
# Load model
# ===============================
model = YOLO("model/best.pt")

# ===============================
# Predict on image
# ===============================
results = model.predict(
    source="images/traffic.png",
    imgsz=640,
    conf=0.25,
    save=False
)

# ===============================
# Draw bounding boxes (small font)
# ===============================
annotated_img = results[0].plot(
    line_width=1,
    font_size=0.45,
    labels=True,
    conf=True
)

# ===============================
# Count objects
# ===============================
object_counts = count_objects(results[0], model)

# ===============================
# Draw object counts (top-left)
# ===============================
y_offset = 30
for obj, count in object_counts.items():
    cv2.putText(
        annotated_img,
        f"{obj}: {count}",
        (15, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )
    y_offset += 25

# ===============================
# Apply alerts
# ===============================
apply_alerts(annotated_img, object_counts)

# ===============================
# Show & save
# ===============================
cv2.imwrite("output/output.jpg", annotated_img)
cv2.imshow("Detection + Alerts", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Image detection completed")
