from ultralytics import YOLO
import cv2
import time
import csv
import os  # ใช้สำหรับเสียงแจ้งเตือน (macOS)

# โหลดโมเดล
model = YOLO("yolov8x.pt")  # เปลี่ยนเป็น yolov8l.pt ได้ถ้าเครื่องช้า

# เปิดวิดีโอ
video_path = "/Users/fourwheel2005/Documents/traffic-ai/video/IMG_63882.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "ไม่สามารถเปิดวิดีโอได้"

# อ่านขนาดวิดีโอจริง
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# สร้างวิดีโอ output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_counted.mp4', fourcc, 30, (width, height))

# กำหนดตำแหน่งเส้นนับ (โซนล่างของภาพ)
count_line_y = int(height * 0.6)
line_thickness = 4

# เตรียมตัวแปร
counted_ids = set()
vehicle_counts = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0}
class_map = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
start_time = time.time()
log = []

# สำหรับแจ้งเตือน
alerted = {k: False for k in vehicle_counts}
thresholds = {"car": 10, "motorbike": 5, "bus": 3, "truck": 3}

# เริ่ม loop ประมวลผล
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

                if abs(center_y - count_line_y) < 25 and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    vehicle_counts[vehicle_type] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label_name} ID:{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)

    # วาดเส้นนับ + โซน
    cv2.line(frame, (0, count_line_y), (width, count_line_y), (0, 255, 255), line_thickness)
    cv2.rectangle(frame, (0, count_line_y - 25), (width, count_line_y + 25), (100, 255, 100), 2)

    # แสดงผลรวม
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Car: {vehicle_counts['car']}", (50, 50), font, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Bike: {vehicle_counts['motorbike']}", (50, 90), font, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Bus: {vehicle_counts['bus']}", (50, 130), font, 1, (0, 200, 255), 2)
    cv2.putText(frame, f"Truck: {vehicle_counts['truck']}", (50, 170), font, 1, (200, 255, 200), 2)

    # ใส่เวลาในวิดีโอ
    frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cv2.putText(frame, f"Time: {frame_time:.1f}s", (width - 250, 50), font, 0.8, (255, 255, 255), 2)

    # แจ้งเตือนถ้ารถเกิน
    for vtype in vehicle_counts:
        if vehicle_counts[vtype] > thresholds[vtype] and not alerted[vtype]:
            msg = f"🚨 {vtype.upper()} overload!"
            y_pos = 50 + 40 * list(vehicle_counts.keys()).index(vtype)
            cv2.putText(frame, msg, (800, y_pos), font, 1, (0, 0, 255), 3)
            os.system(f"say 'Too many {vtype}s detected'")
            alerted[vtype] = True

    # บันทึก log
    elapsed_time = int(time.time() - start_time)
    total = sum(vehicle_counts.values())
    log.append([
        elapsed_time,
        vehicle_counts["car"],
        vehicle_counts["motorbike"],
        vehicle_counts["bus"],
        vehicle_counts["truck"],
        total
    ])

    # แสดงผลและบันทึก
    out.write(frame)
    cv2.imshow("Vehicle Counter with Alert", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดทุกอย่าง
cap.release()
out.release()
cv2.destroyAllWindows()

# เขียนไฟล์ CSV log
with open("vehicle_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time(s)", "Car", "Motorbike", "Bus", "Truck", "Total"])
    writer.writerows(log)

print("✅ เสร็จแล้ว! บันทึกเรียบร้อย:")
print("📄 vehicle_log.csv")
print("🎥 output_counted.mp4")
