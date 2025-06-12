from ultralytics import YOLO
import cv2
import time
import csv

# โหลดโมเดล
model = YOLO("yolov8x.pt")  # ใช้ yolov8l.pt แทนได้ถ้าเครื่องช้า

# เปิดวิดีโอ
video_path = "/Users/fourwheel2005/Documents/traffic-ai/video/IMG_6388.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "ไม่สามารถเปิดวิดีโอได้"

# วิดีโอ Output
width, height = 1280, 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_counted.mp4', fourcc, 30, (width, height))

# พิกัดเส้นนับ (เลื่อนลงให้ตรงเลนล่างของภาพ)
count_line_y = 1500
line_thickness = 4  # เส้นหนาขึ้นมองง่ายขึ้น

# เตรียมข้อมูล
counted_ids = set()
vehicle_counts = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0}
class_map = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
start_time = time.time()
log = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.25,
        imgsz=640,
        iou=0.5,
        tracker="bytetrack.yaml"
    )
    cv2.waitKey(5)

    boxes = results[0].boxes

    if boxes.id is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])
            label = int(box.cls[0])
            label_name = model.names[label]
            center_y = (y1 + y2) // 2

            if label in class_map:
                vehicle_type = class_map[label]

                # ปรับเงื่อนไขให้แม่น: ให้ ±25 px จากเส้นนับ
                if abs(center_y - count_line_y) < 25 and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    vehicle_counts[vehicle_type] += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label_name} ID:{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

    # วาดเส้นนับ
    cv2.line(frame, (0, count_line_y), (width, count_line_y), (0, 255, 255), line_thickness)

    # แสดงผลรวม
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Car: {vehicle_counts['car']}", (50, 50), font, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Bike: {vehicle_counts['motorbike']}", (50, 90), font, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Bus: {vehicle_counts['bus']}", (50, 130), font, 1, (0, 200, 255), 2)
    cv2.putText(frame, f"Truck: {vehicle_counts['truck']}", (50, 170), font, 1, (200, 255, 200), 2)

    # บันทึก log
    elapsed_time = int(time.time() - start_time)
    total = sum(vehicle_counts.values())
    log.append([elapsed_time, vehicle_counts["car"], vehicle_counts["motorbike"],
                vehicle_counts["bus"], vehicle_counts["truck"], total])

    # แสดงและบันทึกวิดีโอ
    out.write(frame)
    cv2.imshow("Vehicle Counter (Fixed)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# บันทึกเป็น CSV
with open("vehicle_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time(s)", "Car", "Motorbike", "Bus", "Truck", "Total"])
    writer.writerows(log)

print("✅ เสร็จแล้ว! นับรถได้ครบ บันทึกไฟล์ออกมาเรียบร้อย:")
print("📄 vehicle_log.csv")
print("🎥 output_counted.mp4")
