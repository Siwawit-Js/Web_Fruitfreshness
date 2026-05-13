import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from data import download_dataset, create_datasets

# กำหนด Path ของโมเดลที่ต้องการโหลด
MODEL_PATH = "fruits_fresh_rotten_baseline.h5"  # หรือ "baseline_model.h5"

def plot_confusion_matrix(cm, class_names):
    """
    วาด Confusion Matrix แบบสวยงามด้วย Seaborn
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # 1. โหลดข้อมูล Test Set
    # (เราต้องการแค่ test_ds แต่ต้องเรียกตาม flow เดิม)
    print("Loading dataset...")
    train_dir, test_dir = download_dataset()
    _, _, test_ds, class_names = create_datasets(train_dir, test_dir)
    
    # 2. โหลดโมเดลที่เทรนเสร็จแล้ว
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 3. เตรียมข้อมูลสำหรับ Scikit-learn
    # เราต้องดึงภาพและ Label ออกมาจาก tf.data.Dataset เพื่อเทียบค่า
    print("Running predictions... (This might take a moment)")
    
    y_true = []
    y_pred_probs = []

    # loop ผ่าน test_ds (ซึ่งปกติจะ batch มา)
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())     # เก็บ label จริง
        y_pred_probs.extend(preds)        # เก็บค่าความน่าจะเป็นที่ทาย
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # แปลงจาก Probability เป็น Class Index (0, 1, 2, ...)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. สร้าง Classification Report (Precision, Recall, F1)
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 5. สร้าง Confusion Matrix
    print("\n" + "="*30)
    print("CONFUSION MATRIX")
    print("="*30)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # 6. Plot กราฟ Confusion Matrix
    plot_confusion_matrix(cm, class_names)

if __name__ == "__main__":
    main()