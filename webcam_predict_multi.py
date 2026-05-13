# webcam_predict.py

import cv2
import numpy as np
import tensorflow as tf

from config import IMG_SIZE, MODEL_SAVE_PATH

# โหลดโมเดล + class names
model = tf.keras.models.load_model(MODEL_SAVE_PATH)
class_names = np.load("class_names.npy")

# ถ้าโหลดมาเป็น array ของ bytes ให้แปลงเป็น str
class_names = [
    name.decode("utf-8") if isinstance(name, bytes) else str(name)
    for name in class_names
]

print("Loaded classes:", class_names)

cap = cv2.VideoCapture(0)

# ลด resolution กล้องลงให้เบาขึ้น (แต่ภาพยังชัดพอ)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

FRAME_SKIP = 3  # predict ทุกๆ 3 เฟรม
frame_count = 0
last_label = "Waiting..."
last_color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถเปิดกล้องได้")
        break

    frame_count += 1

    # ทำการ predict เฉพาะบางเฟรม เพื่อลดโหลด
    if frame_count % FRAME_SKIP == 0:
        # เตรียมภาพให้เข้ากับโมเดล
        img = cv2.resize(frame, IMG_SIZE)
        img = np.expand_dims(img, axis=0) / 255.0

        # predict
        preds = model.predict(img, verbose=0)
        class_id = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_id])
        raw_label = class_names[class_id]

        # mapping label + สี
        label_text = f"{raw_label} ({confidence*100:.1f}%)"

        if "fresh" in raw_label.lower():
            color = (0, 255, 0)       # เขียว
        elif "rotten" in raw_label.lower():
            color = (0, 0, 255)       # แดง
        else:
            color = (0, 255, 255)     # เหลือง

        last_label = label_text
        last_color = color

    # วาดพื้นหลัง + ข้อความ ใช้ last_label จากเฟรมล่าสุดที่ predict แล้ว
    text = last_label
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2

    # คำนวณขนาดกล่องข้อความ
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 20, 40
    # กล่องพื้นหลัง
    cv2.rectangle(
        frame,
        (x - 10, y - text_h - 10),
        (x + text_w + 10, y + baseline + 10),
        (0, 0, 0),
        thickness=-1,
    )
    # ข้อความ
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        last_color,
        thickness,
        cv2.LINE_AA,
    )

    cv2.imshow("Fruit Freshness Scanner - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
