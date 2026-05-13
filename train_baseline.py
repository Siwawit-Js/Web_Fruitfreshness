import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

from config import (
    EPOCHS_HEAD, 
    EPOCHS_FINE,
    BASELINE_CHECKPOINT_PATH,
    BASELINE_MODEL_SAVE_PATH,
)
from data import download_dataset, create_datasets
# เราไม่ใช้ build_base จาก model_builder แล้ว เพราะจะสร้างเอง
from model_builder import compile_model 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_baseline_cnn(num_classes, img_height=224, img_width=224):
    """
    สร้าง Simple CNN Model สำหรับทำ Baseline
    """
    model = models.Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        
        # สำคัญ: ต้อง rescale ค่า pixel จาก [0, 255] เป็น [0, 1] สำหรับโมเดลที่เทรนเอง
        layers.Rescaling(1./255),
        
        # Block 1
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Block 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Block 3
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # ช่วยลด Overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_history(history):
    """
    Plot กราฟสำหรับ Single Phase Training
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, '--', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, '--', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.show()

def main():
    # 1) โหลด dataset
    train_dir, test_dir = download_dataset()
    train_ds, val_ds, test_ds, class_names = create_datasets(train_dir, test_dir)
    num_classes = len(class_names)
    print("Number of classes:", num_classes)

    # 2) สร้างโมเดล Baseline (CNN ธรรมดา)
    print("\nBuilding Baseline CNN Model...")
    model = build_baseline_cnn(num_classes)
    model.summary()

    # 3) Compile โมเดล
    # สำหรับ Baseline เราใช้ Learning Rate มาตรฐาน (เช่น 1e-3 หรือ 1e-4)
    # เราสามารถใช้ฟังก์ชัน compile_model เดิม หรือเขียนใหม่ก็ได้
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4) Callback
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5, 
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            BASELINE_CHECKPOINT_PATH, 
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # 5) เริ่มเทรน (Training Phase เดียว)
    # รวม epoch ทั้งหมดเข้าด้วยกัน หรือกำหนดใหม่
    TOTAL_EPOCHS = EPOCHS_HEAD + EPOCHS_FINE 
    
    print(f"\n===== Start Training Baseline (Total Epochs: {TOTAL_EPOCHS}) =====")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs= 5,
        callbacks=callbacks,
    )

    # 6) Plot กราฟ
    plot_history(history)

    # 7) ประเมินผลบน test set
    print("\nEvaluating on Test Set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # 8) เซฟโมเดล
    model.save(BASELINE_MODEL_SAVE_PATH)
    print(f"Model saved to {BASELINE_MODEL_SAVE_PATH}")

    # 9) เซฟ class_names
    np.save("class_names.npy", np.array(class_names))
    print("class_names saved to class_names.npy")

if __name__ == "__main__":
    main()