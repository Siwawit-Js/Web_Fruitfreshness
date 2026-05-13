# server.py

import io
import cv2
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from config import IMG_SIZE, MODEL_SAVE_PATH

# ---------- โหลดโมเดล ----------
model = tf.keras.models.load_model(MODEL_SAVE_PATH)
class_names = np.load("class_names.npy")

class_names = [
    name.decode("utf-8") if isinstance(name, bytes) else str(name)
    for name in class_names
]
print("Loaded classes:", class_names)

# ---------- สร้าง FastAPI app ----------
app = FastAPI()

# ---------- เปิด CORS ให้ web ที่รันจาก 127.0.0.1:5500 / localhost:5500 ----------
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5501",
    "http://localhost:5501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # ถ้า懶จะใช้ origins เฉพาะ ก็ใช้ ["*"] ได้
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ฟังก์ชันช่วยเตรียมภาพ ----------
def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("ไม่สามารถอ่านรูปภาพได้")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    return img


# ---------- /predict : ใช้กับหน้า camera.html แบบเดิม ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = preprocess_image_from_bytes(contents)

        preds = model.predict(img, verbose=0)
        class_id = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_id])
        label = class_names[class_id]

        return {
            "label": label,
            "confidence": confidence,
        }
    except ValueError as e:
        # รูปผิดรูปแบบ
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # log ข้อผิดพลาดไว้ใน console
        print("ERROR /predict:", e)
        raise HTTPException(status_code=500, detail="Internal error while predicting")


# ---------- ถ้าอยากใช้แบบหลายลูก /detect_multi (เว็บ realtime ตัวใหม่) ----------
# ---------- ถ้าอยากใช้แบบหลายลูก /detect_multi (เว็บ realtime ตัวใหม่) ----------
@app.post("/detect_multi")
async def detect_multi(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("ไม่สามารถอ่านรูปภาพได้")

        # แปลงเป็น RGB สำหรับโมเดล
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ทำ threshold หา object แบบง่าย ๆ
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thresh = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = frame.shape[:2]

        # -------------------------
        # เลือกแค่ contour ที่ "ใหญ่ที่สุด"
        # -------------------------
        best_cnt = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:      # ตัด noise เล็ก ๆ ทิ้งไป
                continue
            if area > best_area:
                best_area = area
                best_cnt = cnt

        objects = []

        if best_cnt is not None:
            x, y, bw, bh = cv2.boundingRect(best_cnt)
            pad = 10
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + bw + pad, w)
            y2 = min(y + bh + pad, h)

            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size > 0:
                img = cv2.resize(crop, IMG_SIZE)
                # ❗ ไม่ต้องหาร 255 ให้เหมือนตอนเทรน (ใช้ preprocess_input ในโมเดลแล้ว)
                img = img.astype("float32")
                img = np.expand_dims(img, axis=0)

                preds = model.predict(img, verbose=0)
                class_id = int(np.argmax(preds[0]))
                conf = float(preds[0][class_id])
                label = class_names[class_id].lower()

                if "fresh" in label:
                    status = "fresh"
                    fruit_name = label.replace("fresh", "").strip()
                elif "rotten" in label:
                    status = "rotten"
                    fruit_name = label.replace("rotten", "").strip()
                else:
                    status = "unknown"
                    fruit_name = label

                objects.append(
                    {
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "label": class_names[class_id],
                        "fruit_name": fruit_name,
                        "status": status,
                        "confidence": conf,
                    }
                )

        return {"objects": objects}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("ERROR /detect_multi:", e)
        raise HTTPException(status_code=500, detail="Internal error while detecting")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
