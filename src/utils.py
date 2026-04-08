from typing import Any

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def apply_clahe(
    image: np.ndarray,
    grid_size: int = 8,
    clip_limit: float = 2.0,
    is_training: bool = False,
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to each channel
    """
    if is_training:
        grid_size = int(
            grid_size + np.random.uniform(-np.log2(grid_size), np.log2(grid_size))
        )
        clip_limit = clip_limit + np.random.uniform(
            -np.log2(clip_limit), np.log2(clip_limit)
        )

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    channels = cv2.split(image)
    equalized_channels = [clahe.apply(channel) for channel in channels]

    return cv2.merge(equalized_channels)


def preprocess_image(
    image_path: str, laterality: str, target_size: tuple = (416, 320)
) -> np.ndarray:
    """
    Preprocesses the image by loading, orienting, downsampling, and expanding to 3 channels.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(image_path)

    # Orient the image (breast left, nipple right)
    image = cv2.flip(image, 1) if laterality.upper() == "R" else image

    # Downsample the image using lanczos interpolation
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Expand the image to 3 channels
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


class MammoCNNDataset(Dataset):
    """
    Dataset clss for first stage of MAMMO CNN training
    """
    def __init__(self, image_df: pd.DataFrame, is_training: bool = False) -> None:
        self.image_df = image_df
        self.is_training = is_training

    def __len__(self) -> int:
        return self.image_df.shape[0]

    def __getitem__(self, index: int) -> dict:
        row = self.image_df.iloc[index]
        image_path = row["image_path"]
        laterality = row["laterality"]
        diagnosis = row["diagnosis"]
        sign = row["finding_categories"]
        suspicion = row["breast_birads"]
        density = row["breast_density"]
        age = row["Patient's Age"]

        image = preprocess_image(image_path, laterality)
        image = apply_clahe(image, is_training=self.is_training)

        if self.is_training:
            # Apply random gaussian noise with σ=0.01 per channel
            image = image.astype(np.float32) / 255.0
            noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        # Standardize the image per channel (mean = 0, std = 1)
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (std + 1e-8)

        return {
            "mammogram": image,
            "diagnosis": diagnosis,
            "sign": sign,
            "suspicion": suspicion,
            "density": density,
            "age": age,
        }
