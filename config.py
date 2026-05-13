# config.py

# ขนาดภาพที่ใช้เทรน / infer
IMG_SIZE = (224, 224)

# batch size ตอนเทรน
BATCH_SIZE = 32

# เทรน 2 เฟส
EPOCHS_HEAD = 8        # เฟสแรก: freeze base แล้วเทรนเฉพาะหัว
EPOCHS_FINE = 12       # เฟสสอง: fine-tune บางชั้นบนของ MobileNetV2

# path สำหรับ checkpoint + โมเดลสุดท้าย
CHECKPOINT_PATH = "best_fruit_model.keras"
MODEL_SAVE_PATH = "fruits_fresh_rotten_mobilenetv2.h5"

BASELINE_CHECKPOINT_PATH = "baseline_fruit_model.keras"
BASELINE_MODEL_SAVE_PATH = "fruits_fresh_rotten_baseline.h5"

# Kaggle dataset id
KAGGLE_DATASET_ID = "sriramr/fruits-fresh-and-rotten-for-classification"
