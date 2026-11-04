from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch


MODEL_PATH = Path(__file__).resolve().parent / 'static/models/model.pth'
IMAGE_SIZE = 256
THRESHOLD = 0.5


@lru_cache(maxsize=1)
def load_model():
    model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.inference_mode()
def segment_image(image_path, alpha=0.75):
    if not 0 <= alpha <= 1:
        raise ValueError('Alpha must be between 0 and 1.')

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if original is None or gray is None:
        raise ValueError('Unable to read the image.')

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    height, width = gray.shape

    image = cv2.resize(
        gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    image = image.astype(np.float32) / 127.5 - 1.0
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    model = load_model()
    logits = model(input_tensor)
    probabilities = torch.sigmoid(logits)
    mask_small = (
        probabilities[0, 0].cpu().numpy() >= THRESHOLD
    ).astype(np.uint8)
    mask = cv2.resize(
        mask_small, (width, height), interpolation=cv2.INTER_NEAREST
    )

    color_mask = np.zeros_like(original)
    color_mask[mask == 1] = [0, 255, 0]
    overlay = original.copy()
    overlay[mask == 1] = (
        original[mask == 1] * (1 - alpha)
        + color_mask[mask == 1] * alpha
    ).astype(np.uint8)

    return mask * 255, overlay
