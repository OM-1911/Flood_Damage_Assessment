"""Train the U-Net flood segmentation model from dataset/Image and dataset/Mask."""
from pathlib import Path
import os

# CPU-optimized execution
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '1')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / 'dataset' / 'Image'
MASK_DIR = BASE_DIR / 'dataset' / 'Mask'
MODEL_DIR = BASE_DIR / 'model'
MODEL_PATH = MODEL_DIR / 'flood_unet.keras'
PLOT_DIR = MODEL_DIR / 'training_plots'

IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 8
SEED = 42


def load_dataset():
    images, masks = [], []
    files = sorted(IMAGE_DIR.glob('*.jpg'), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    for image_path in files:
        mask_path = MASK_DIR / f'{image_path.stem}.png'
        if not mask_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        images.append(image.astype(np.float32) / 255.0)
        masks.append((mask > 127).astype(np.float32))

    if not images:
        raise RuntimeError('No matching Image/*.jpg and Mask/*.png pairs found.')
    X = np.asarray(images, dtype=np.float32)
    Y = np.asarray(masks, dtype=np.float32)[..., None]
    print(f'Loaded {len(X)} image/mask pairs.')
    print('Images:', X.shape, 'Masks:', Y.shape)
    return X, Y

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x


@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=[1, 2, 3])
    total = tf.reduce_sum(y_true_f, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_f, axis=[1, 2, 3])
    return tf.reduce_mean((2.0 * intersection + smooth) / (total + smooth))


@tf.keras.utils.register_keras_serializable()
def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=[1, 2, 3])
    total = tf.reduce_sum(y_true_f, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_f, axis=[1, 2, 3])
    union = total - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


@tf.keras.utils.register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return tf.reduce_mean(bce) + dice


def build_unet():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    # Encoder
    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D()(c3)
    
    # Bridge
    bridge = conv_block(p3, 256)
    bridge = layers.Dropout(0.2)(bridge)
    
    # Decoder
    u3 = layers.UpSampling2D()(bridge); u3 = layers.Concatenate()([u3, c3]); c4 = conv_block(u3, 128)
    u2 = layers.UpSampling2D()(c4); u2 = layers.Concatenate()([u2, c2]); c5 = conv_block(u2, 64)
    u1 = layers.UpSampling2D()(c5); u1 = layers.Concatenate()([u1, c1]); c6 = conv_block(u1, 32)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c6)
    return models.Model(inputs, outputs, name='FloodUNet')


def plot_history(history):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Loss plot
    if 'loss' in history.history:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Training Loss', color='#087ea4', lw=2)
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss', color='#e63946', lw=2)
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('U-Net BCE + Dice Loss'); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(PLOT_DIR / 'loss.png', dpi=150); plt.close()

    # Dice plot
    if 'dice_coefficient' in history.history:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['dice_coefficient'], label='Training Dice', color='#2a9d8f', lw=2)
        if 'val_dice_coefficient' in history.history:
            plt.plot(history.history['val_dice_coefficient'], label='Validation Dice', color='#e76f51', lw=2)
        plt.xlabel('Epoch'); plt.ylabel('Dice Score'); plt.title('Dice Coefficient'); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(PLOT_DIR / 'dice.png', dpi=150); plt.close()

    # Accuracy plot
    if 'accuracy' in history.history:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='#087ea4', lw=2)
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#2a9d8f', lw=2)
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Segmentation Accuracy'); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(PLOT_DIR / 'accuracy.png', dpi=150); plt.close()


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    X, Y = load_dataset()
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.20, random_state=SEED)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=SEED)
    print(f'Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}')

    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss=bce_dice_loss,
        metrics=['accuracy', dice_coefficient, iou_score]
    )
    model.summary()

    MODEL_DIR.mkdir(exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    # Pre-cache training dataset for fast training
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).cache().shuffle(len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    plot_history(history)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(BATCH_SIZE)
    results = model.evaluate(test_ds, verbose=0, return_dict=True)
    print('\nTest results:')
    for key, value in results.items():
        print(f'{key}: {value:.4f}')

    model.save(MODEL_PATH)
    print(f'\nModel saved to: {MODEL_PATH}')
    print(f'Training graphs saved to: {PLOT_DIR}')


if __name__ == '__main__':
    main()
