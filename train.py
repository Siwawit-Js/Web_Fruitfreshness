# train.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from config import (
    EPOCHS_HEAD,
    EPOCHS_FINE,
    CHECKPOINT_PATH,
    MODEL_SAVE_PATH,
)
from data import download_dataset, create_datasets
from model_builder import build_base, compile_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def plot_history(histories, labels):
    """
    histories: list ของ History object
    labels:   list ของชื่อเฟส เช่น ["head", "fine-tune"]
    """
    plt.figure(figsize=(12, 4))

    # ---- Accuracy ----
    plt.subplot(1, 2, 1)
    for h, name in zip(histories, labels):
        acc = h.history["accuracy"]
        val_acc = h.history["val_accuracy"]
        epochs_range = range(1, len(acc) + 1)
        plt.plot(epochs_range, acc, label=f"{name} - train")
        plt.plot(epochs_range, val_acc, "--", label=f"{name} - val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")

    # ---- Loss ----
    plt.subplot(1, 2, 2)
    for h, name in zip(histories, labels):
        loss = h.history["loss"]
        val_loss = h.history["val_loss"]
        epochs_range = range(1, len(loss) + 1)
        plt.plot(epochs_range, loss, label=f"{name} - train")
        plt.plot(epochs_range, val_loss, "--", label=f"{name} - val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    plt.tight_layout()
    plt.show()


def main():
    # 1) โหลด dataset
    train_dir, test_dir = download_dataset()
    train_ds, val_ds, test_ds, class_names = create_datasets(train_dir, test_dir)
    num_classes = len(class_names)
    print("Number of classes:", num_classes)

    # 2) สร้างโมเดล base + compile
    model, base_model = build_base(num_classes)
    model.summary()

    # 3) callback
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # -------------------------
    # เฟสที่ 1: เทรนเฉพาะหัว (base_model ไม่ train)
    # -------------------------
    base_model.trainable = False
    compile_model(model, lr=1e-4)

    print("\n===== Phase 1: Train head only =====")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    # -------------------------
    # เฟสที่ 2: Fine-tune บางส่วนของ base_model
    # -------------------------
    print("\n===== Phase 2: Fine-tune last blocks =====")
    # ปลดล็อกประมาณครึ่งบนของ MobileNetV2
    fine_tune_at = len(base_model.layers) // 2
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    compile_model(model, lr=1e-5)

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=callbacks,
    )

    # รวม history ไว้ plot
    plot_history(
        histories=[history_head, history_fine],
        labels=["head", "fine-tune"],
    )

    # 4) ประเมินผลบน test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # 5) เซฟโมเดล
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # 6) เซฟ class_names ไว้ใช้ตอน predict (server.py)
    np.save("class_names.npy", np.array(class_names))
    print("class_names saved to class_names.npy")


if __name__ == "__main__":
    main()
