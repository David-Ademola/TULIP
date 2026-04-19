import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


def apply_clahe(
    image: np.ndarray | torch.Tensor,
    grid_size: int = 8,
    clip_limit: float = 2.0,
    is_training: bool = False,
) -> torch.Tensor:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to each channel. Accepts either a NumPy array or a Torch tensor and
    returns a Torch tensor in CHW format.
    """
    # Accept both NumPy arrays and torch tensors
    tensor_input = isinstance(image, torch.Tensor)

    if tensor_input:
        if image.ndim == 3:
            # Convert CHW -> HWC
            image_np = image.permute(1, 2, 0).cpu().numpy()
        elif image.ndim == 2:
            image_np = image.cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor image shape: {image.shape}")
    else:
        image_np = image

    if image_np.dtype in (np.float32, np.float64, np.float16):
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    if image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[-1] == 1):
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.ndim == 3 and image_np.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape for CLAHE: {image_np.shape}")

    if is_training:
        grid_size = int(
            grid_size + np.random.uniform(-np.log2(grid_size), np.log2(grid_size))
        )
        clip_limit = clip_limit + np.random.uniform(
            -np.log2(clip_limit), np.log2(clip_limit)
        )

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    channels = cv2.split(image_np)
    equalized_channels = [clahe.apply(channel) for channel in channels]
    image_np = cv2.merge(equalized_channels)

    image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1)

    return image_tensor


def preprocess_image(
    image_path: str, laterality: str, target_size: tuple = (416, 320)
) -> torch.Tensor:
    """
    Preprocesses the image by loading, orienting, downsampling, and
    returning a normalized grayscale tensor.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(image_path)

    image = cv2.flip(image, 1) if laterality.upper() == "R" else image
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

    image = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

    return image


class MammoCNNDataset(Dataset):
    """
    Dataset class for first stage of MAMMO CNN training
    """

    def __init__(
        self,
        image_df: pd.DataFrame,
        transform: Compose | None = None,
        is_training: bool = False,
    ) -> None:
        self.image_df = image_df
        self.transform = transform
        self.is_training = is_training

    def __len__(self) -> int:
        return self.image_df.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.image_df.iloc[index]
        image_path = row["image_path"]
        laterality = row["laterality"]
        diagnosis = row["diagnosis"]
        findings = row["finding_categories"]
        suspicion = row["breast_birads"]
        density = row["breast_density"]
        age = row["age"]

        image = preprocess_image(image_path, laterality)

        # Apply data augmentations
        if self.transform:
            image = self.transform(image)

        image = apply_clahe(image, is_training=self.is_training)

        # Apply random gaussian noise during training
        if self.is_training:
            image = image.float()
            noise = torch.randn_like(image) * 0.01
            image = torch.clamp(image + noise, 0.0, 1.0)

        # Standardize the image
        image = image.float()
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (std + 1e-8)

        return {
            "mammogram": image,
            "diagnosis": torch.tensor(diagnosis, dtype=torch.long),
            "findings": torch.tensor(findings, dtype=torch.float32),  # [10]
            "suspicion": torch.tensor(suspicion, dtype=torch.float32),  # [5]
            "density": torch.tensor(density, dtype=torch.float32),  # [4]
            "age": torch.tensor(age, dtype=torch.float32),
        }
