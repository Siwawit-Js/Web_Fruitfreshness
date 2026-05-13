# 🍎 Fruit Freshness Detector

ระบบตรวจจับ "ผลไม้สด / ผลไม้เน่า" แบบ Realtime ด้วย Deep Learning
สำหรับวิชา **CS461 - Final Project**

โปรเจกต์นี้ใช้ **MobileNetV2 (Transfer Learning)** เปรียบเทียบกับโมเดล **Baseline CNN**
ที่เทรนเองตั้งแต่ต้น โดยรับภาพจากกล้องเว็บแคมแล้วส่งไปทำนายผลผ่าน FastAPI Backend
และแสดงผลทันทีบนหน้าเว็บ

---

## 📂 โครงสร้างโปรเจกต์

```
Fruit freshness/
├── camera.html              # หน้าเว็บแสดงผลกล้องและผลการทำนายแบบ realtime
├── server.py                # FastAPI server เปิด endpoint /predict, /detect_multi
├── config.py                # ค่าคงที่ของโปรเจกต์ (image size, batch, path โมเดล)
├── data.py                  # โหลด dataset จาก Kaggle + สร้าง tf.data pipeline
├── model_builder.py         # โครงสร้าง MobileNetV2 + data augmentation
├── train.py                 # เทรนโมเดลหลัก MobileNetV2 (2 เฟส: head + fine-tune)
├── train_baseline.py        # เทรนโมเดล Baseline CNN เพื่อเปรียบเทียบ
├── eval.py                  # ประเมินผลโมเดล (classification report + confusion matrix)
├── webcam_predict_multi.py  # ทดสอบโมเดลผ่าน OpenCV (กล้อง local โดยไม่ใช้เว็บ)
├── class_names.npy          # ชื่อ class ที่บันทึกตอนเทรน (โหลดตอน inference)
├── requirements.txt         # รายการ Python dependencies
└── README.md
```

> ⚠️ ไฟล์โมเดล (`*.h5`, `*.keras`) ไม่ได้ถูกอัพโหลดขึ้น GitHub เนื่องจากมีขนาด ~150 MB
> เกินขีดจำกัด 100 MB ของ GitHub กรุณาเทรนใหม่ตามขั้นตอนด้านล่าง

---

## 🧠 Dataset

ใช้ dataset **Fruits Fresh and Rotten for Classification** จาก Kaggle
[https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)

มี 6 class:

| Class | คำอธิบาย |
|---|---|
| freshapples | แอปเปิ้ลสด |
| freshbanana | กล้วยสด |
| freshoranges | ส้มสด |
| rottenapples | แอปเปิ้ลเน่า |
| rottenbanana | กล้วยเน่า |
| rottenoranges | ส้มเน่า |

dataset จะถูกโหลดอัตโนมัติผ่าน `kagglehub` เมื่อรัน `train.py`

---

## ⚙️ Setup & Installation

### 0. ติดตั้ง Python 3.10

ดาวน์โหลดและติดตั้ง Python 3.10 ได้จากเว็บไซต์ทางการ
[https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)

หลังติดตั้งตรวจสอบเวอร์ชันด้วย:

```bash
python --version
```

### 1. Clone โปรเจกต์

```bash
git clone https://github.com/Siwawit-Js/Web_Fruitfreshness.git
cd Web_Fruitfreshness
```

### 2. สร้างและเปิดใช้งาน Virtual Environment

```bash
python -m venv tfenv
```

**Windows:**
```bash
tfenv\Scripts\activate
```

**macOS / Linux:**
```bash
source tfenv/bin/activate
```

### 3. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

หาก pip timeout ระหว่างติดตั้ง ให้ใช้:

```bash
pip install --default-timeout=1000 -r requirements.txt
```

---

## 🚀 การใช้งาน

### ขั้นตอนที่ 1: เทรนโมเดล (รันครั้งแรกครั้งเดียว)

เนื่องจากไฟล์โมเดลไม่ได้ถูกอัพโหลดมา ต้องเทรนใหม่ก่อน:

**โมเดลหลัก (MobileNetV2 Transfer Learning):**
```bash
python train.py
```

**โมเดล Baseline (CNN ธรรมดา) — ทางเลือกสำหรับเปรียบเทียบ:**
```bash
python train_baseline.py
```

หลังเทรนเสร็จจะได้ไฟล์:
- `fruits_fresh_rotten_mobilenetv2.h5` (โมเดลหลักที่ใช้ใน server.py)
- `best_fruit_model.keras` (checkpoint ระหว่างเทรน)
- `class_names.npy` (รายชื่อ class)

### ขั้นตอนที่ 2: รัน Backend Server

```bash
python server.py
```

Server จะรันที่ `http://127.0.0.1:8000` พร้อม endpoint:
- `POST /predict` — รับรูปภาพ → คืนค่า label + confidence
- `POST /detect_multi` — ตรวจจับวัตถุที่ใหญ่ที่สุดในเฟรมแล้วทำนาย

### ขั้นตอนที่ 3: เปิดหน้าเว็บ

**วิธีที่ 1: ใช้ Live Server บน VS Code**

คลิกขวาที่ `camera.html` → `Open with Live Server`

**วิธีที่ 2: รันผ่าน Python http.server**

เปิด Terminal ใหม่ในโฟลเดอร์โปรเจกต์ แล้วรัน:
```bash
python -m http.server 8000
```

จากนั้นเปิดเบราว์เซอร์ไปที่ [http://localhost:8000/camera.html](http://localhost:8000/camera.html)

> 💡 หน้าเว็บจะขออนุญาตใช้กล้อง → กดอนุญาต แล้วระบบจะส่งภาพไปทำนายอัตโนมัติทุก 0.7 วินาที
> - กรอบสีเขียว = ผลไม้สด
> - กรอบสีแดง = ผลไม้เน่า

---

## 📊 การประเมินผลโมเดล

ประเมินผล classification report + confusion matrix:

```bash
python eval.py
```

> สามารถเปลี่ยน `MODEL_PATH` ใน `eval.py` เพื่อสลับระหว่างโมเดล Baseline กับ MobileNetV2

---

## 🧪 ทดสอบผ่าน OpenCV โดยตรง (ไม่ใช้เว็บ)

```bash
python webcam_predict_multi.py
```

กด `Q` เพื่อปิดหน้าต่าง

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Deep Learning**: TensorFlow 2.10 (Keras), MobileNetV2 (ImageNet pre-trained)
- **Computer Vision**: OpenCV
- **Frontend**: HTML5 + Vanilla JavaScript (getUserMedia API)
- **Dataset**: Kaggle Hub

---

## 👤 ผู้พัฒนา

- **Siwawit-Js** — [GitHub](https://github.com/Siwawit-Js)

โปรเจกต์ในรายวิชา **CS461** — Final Project
