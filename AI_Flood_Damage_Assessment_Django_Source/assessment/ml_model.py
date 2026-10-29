import os
import threading
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings

IMG_SIZE = 128
MODEL_PATH = Path(settings.BASE_DIR) / 'model' / 'flood_unet.keras'

_model = None
_infer_fn = None
_lock = threading.Lock()
_ready_event = threading.Event()
_is_warming_up = False


def _load_model_internal():
    global _model, _infer_fn
    if _model is not None and _infer_fn is not None:
        return _model, _infer_fn

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Model not found at {MODEL_PATH}. Run "python train_model.py" first.'
        )

    # Configure environment flags for silent, optimized CPU inference
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '1')

    import tensorflow as tf

    # Load model architecture and weights without compilation overhead
    loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Compile a direct callable graph for 2x-5x faster single-image inference
    @tf.function(reduce_retracing=True)
    def _fast_predict(tensor):
        return loaded_model(tensor, training=False)

    # Warm up graph execution with dummy forward pass so user requests don't lag
    dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    _ = _fast_predict(dummy_input)

    _model = loaded_model
    _infer_fn = _fast_predict
    _ready_event.set()
    return _model, _infer_fn


def get_model():
    """Thread-safe retrieval of the loaded model and inference function."""
    if _ready_event.is_set():
        return _model, _infer_fn

    with _lock:
        if _model is not None and _infer_fn is not None:
            return _model, _infer_fn
        return _load_model_internal()


def warm_up_in_background():
    """Asynchronously loads and warms up the model in a background daemon thread."""
    global _is_warming_up
    with _lock:
        if _model is not None or _is_warming_up:
            return
        _is_warming_up = True

    def _worker():
        try:
            with _lock:
                _load_model_internal()
        except Exception:
            pass

    thread = threading.Thread(target=_worker, name="FloodAI-ModelPreloader", daemon=True)
    thread.start()


def _severity(percent):
    if percent < 10:
        return 'Low'
    if percent < 30:
        return 'Moderate'
    if percent < 60:
        return 'High'
    return 'Severe'


def predict_flood(image_path):
    _, infer_fn = get_model()
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError('The uploaded file is not a valid image.')

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    height, width = original.shape[:2]

    resized = cv2.resize(original_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    # Fast compiled inference
    prediction = infer_fn(x).numpy()[0, :, :, 0]
    small_mask = (prediction >= 0.5).astype(np.uint8)
    mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    flooded_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    flood_percentage = (flooded_pixels / total_pixels) * 100.0
    severity = _severity(flood_percentage)

    # Calculate model prediction confidence / certainty score (0% - 100%)
    pixel_confidence = np.maximum(prediction, 1.0 - prediction)
    accuracy = round(float(np.mean(pixel_confidence) * 100.0), 2)

    # High-contrast visual segmentation mask:
    # Dry land / non-flooded: Dark charcoal/navy (BGR: [42, 23, 15])
    # Flooded water: Glowing cyan/blue (BGR: [255, 210, 0])
    visual_mask = np.zeros((height, width, 3), dtype=np.uint8)
    visual_mask[:] = [42, 23, 15]
    flood_bool = (mask == 1)
    if np.any(flood_bool):
        visual_mask[flood_bool] = [255, 210, 0]
        # Outline boundary between flood and dry land in crisp white
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visual_mask, contours, -1, (255, 255, 255), 1)

    # Vectorized in-place overlay without redundant zero-array allocation
    overlay = original.copy()
    if np.any(flood_bool):
        pts = overlay[flood_bool].astype(np.float32)
        pts[:, 0] *= 0.40  # Blue
        pts[:, 1] *= 0.40  # Green
        pts[:, 2] = pts[:, 2] * 0.40 + 153.0  # Red (255 * 0.60)
        overlay[flood_bool] = pts.astype(np.uint8)

    return mask, visual_mask, overlay, round(flood_percentage, 2), severity, accuracy
