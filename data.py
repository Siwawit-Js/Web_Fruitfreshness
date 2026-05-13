# data.py

import os
import kagglehub
import tensorflow as tf
from config import IMG_SIZE, BATCH_SIZE, KAGGLE_DATASET_ID

AUTOTUNE = tf.data.AUTOTUNE


def download_dataset():
    """
    ดาวน์โหลด dataset จาก Kaggle แล้วคืนค่า path train_dir, test_dir

    บางเครื่องโครงสร้างโฟลเดอร์ของ fruits-fresh-and-rotten-for-classification
    จะเป็นแบบ:

    path/
      train/
      test/

    แต่บางครั้งจะเป็นแบบ:
      path/dataset/train
      path/dataset/test

    หรือ:
      path/dataset/dataset/train
      path/dataset/dataset/test

    ฟังก์ชันนี้จะลองเช็คหลายแบบให้อัตโนมัติ
    """
    path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    print("Dataset path:", path)

    # candidate layout หลายแบบ
    candidate_roots = [
        path,
        os.path.join(path, "dataset"),
        os.path.join(path, "dataset", "dataset"),
    ]

    train_dir = None
    test_dir = None

    for root in candidate_roots:
        tdir = os.path.join(root, "train")
        vdir = os.path.join(root, "test")
        if os.path.isdir(tdir) and os.path.isdir(vdir):
            train_dir = tdir
            test_dir = vdir
            break

    if train_dir is None or test_dir is None:
        raise RuntimeError("ไม่พบโฟลเดอร์ train/test ใน dataset path: " + path)

    print("Using train_dir:", train_dir)
    print("Using test_dir :", test_dir)
    return train_dir, test_dir


def create_datasets(train_dir, test_dir, val_ratio=0.2, seed=42):
    """
    สร้าง tf.data.Dataset: train_ds, val_ds, test_ds, class_names
    """

    train_full_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    class_names = train_full_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names, " (", num_classes, " classes )")

        # นับจำนวน batch ทั้งหมด แล้วแปลงเป็น int ของ Python
    train_batches = tf.data.experimental.cardinality(train_full_ds).numpy()
    train_batches = int(train_batches)

    # จำนวน batch ที่ใช้เป็น validation
    val_batches = int(train_batches * val_ratio)


    val_ds = train_full_ds.take(val_batches)
    train_ds = train_full_ds.skip(val_batches)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    def prepare(ds, shuffle=False):
        if shuffle:
            ds = ds.shuffle(1000, seed=seed)
        return ds.cache().prefetch(AUTOTUNE)

    train_ds = prepare(train_ds, shuffle=True)
    val_ds = prepare(val_ds, shuffle=False)
    test_ds = prepare(test_ds, shuffle=False)

    return train_ds, val_ds, test_ds, class_names
