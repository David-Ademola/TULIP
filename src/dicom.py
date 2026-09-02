"""
DICOM -> 16-bit PNG conversion for VinDr-Mammo.
"""

import multiprocessing
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pydicom import dcmread
from pydicom.pixels import apply_voi_lut  # type: ignore
from tqdm.auto import tqdm

# Storage resolution, (height, width).
STORE_SIZE: tuple[int, int] = (1920, 1536)
PNG_COMPRESSION: int = 3
_INVERTED: str = "MONOCHROME1"


def dicom_to_array(
    dicom_path: str | Path,
    laterality: str,
    target_size: tuple[int, int] = STORE_SIZE,
) -> np.ndarray:
    """
    Read one DICOM and return a uint16 array of exactly `target_size`.

    The image is NOT cropped -- the paper states "mammograms were not
    cropped", so breast-region cropping stays a separate later ablation.

    Args:
        dicom_path: path to the .dicom file
        laterality: "L" or "R"; decides which side the padding goes on
        target_size: (height, width) of the returned array

    Returns:
        uint16 array of shape `target_size`, full 0-65535 range, breast tissue
        bright, zero padding on the bottom and on the non-breast side.
    """
    dataset = dcmread(dicom_path)

    slope = float(dataset.get("RescaleSlope", 1))
    intercept = float(dataset.get("RescaleIntercept", 0))
    assert (slope, intercept) == (1.0, 0.0), (
        f"{dicom_path}: non-identity modality LUT "
        f"(slope={slope}, intercept={intercept}); apply_modality_lut is needed"
    )

    # Apply the VOI transform
    # Note the return dtype is NOT stable: uint16 for the LUT-sequence path,
    # float64 for the windowing path. Hence the explicit float32 cast below.
    array = apply_voi_lut(dataset.pixel_array, dataset)

    # Normalise by the stored bit depth, not by 65535.
    bits_stored = int(dataset.BitsStored)
    array = array.astype(np.float32) / float(2**bits_stored - 1)

    if dataset.PhotometricInterpretation == _INVERTED:
        array = 1.0 - array

    array = np.clip(array, 0.0, 1.0)

    target_height, target_width = target_size
    height, width = array.shape

    # Resize preserving aspect ratio, then pad -- never squash.
    scale = min(target_height / height, target_width / width)
    new_height = min(round(height * scale), target_height)
    new_width = min(round(width * scale), target_width)
    array = cv2.resize(array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    array = np.clip(array, 0.0, 1.0)

    # Pad on the side away from the breast, so that after the Dataset's
    # existing laterality flip (preprocess_image flips R) the breast is pinned
    # to the left edge with all padding on the right and bottom.
    pad_bottom = target_height - new_height
    pad_x = target_width - new_width
    is_right = laterality.upper() == "R"
    array = cv2.copyMakeBorder(
        array,
        0,
        pad_bottom,
        pad_x if is_right else 0,
        0 if is_right else pad_x,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    array = (array * 65535.0).round().astype(np.uint16)

    assert array.shape == target_size, f"{dicom_path}: got {array.shape}"
    assert array.dtype == np.uint16, f"{dicom_path}: got {array.dtype}"

    return array


def convert_one(job: tuple[str, str, str, tuple[int, int]]) -> tuple[str, str | None]:
    """
    Convert a single DICOM to a 16-bit PNG. Top-level and tuple-argument so it
    is picklable by multiprocessing.Pool on any start method.

    Args:
        job: (dicom_path, png_path, laterality, target_size)

    Returns:
        (png_path, None) on success, or (png_path, error message) on failure.
        Failures are returned rather than raised so one bad file cannot abort
        a 20,000-image run.
    """
    dicom_path, png_path, laterality, target_size = job

    try:
        array = dicom_to_array(dicom_path, laterality, target_size)

        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        written = cv2.imwrite(
            png_path, array, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
        )
        if not written:
            return png_path, "cv2.imwrite returned False"

        return png_path, None
    # pylint: disable = W0718
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        return png_path, f"{type(error).__name__}: {error}"


def build_jobs(
    metadata_df: pd.DataFrame,
    dicom_root: str | Path,
    png_root: str | Path,
    target_size: tuple[int, int] = STORE_SIZE,
    skip_existing: bool = True,
) -> list[tuple[str, str, str, tuple[int, int]]]:
    """
    Build the conversion job list, mirroring the DICOM tree's
    <study_id>/<image_id> layout into `png_root`.

    `skip_existing=True` makes the run resumable.
    """
    jobs = []

    for row in metadata_df.itertuples(index=False):
        dicom_path = os.path.join(dicom_root, row.study_id, f"{row.image_id}.dicom")  # type: ignore
        png_path = os.path.join(png_root, row.study_id, f"{row.image_id}.png")  # type: ignore

        if skip_existing and os.path.exists(png_path):
            continue

        jobs.append((str(dicom_path), str(png_path), row.laterality, target_size))

    return jobs


def convert_dataset(
    metadata_df: pd.DataFrame,
    dicom_root: str | Path,
    png_root: str | Path,
    target_size: tuple[int, int] = STORE_SIZE,
    workers: int | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """
    Convert every image in `metadata_df` to a 16-bit PNG under `png_root`.

    Returns a summary dict with the converted count, the skipped count and the
    list of failures. Measured ~0.12 s/image single-process, so ~40 min for
    20,000 images serially and a few minutes across cores.
    """
    jobs = build_jobs(metadata_df, dicom_root, png_root, target_size, skip_existing)
    skipped = len(metadata_df) - len(jobs)

    if workers is None:
        workers = max(1, (multiprocessing.cpu_count() or 2) - 1)

    failures: list[tuple[str, str]] = []

    if not jobs:
        return {"converted": 0, "skipped": skipped, "failures": failures}

    if workers == 1:
        results = (convert_one(job) for job in jobs)
        for png_path, error in tqdm(results, total=len(jobs), desc="DICOM->PNG"):
            if error:
                failures.append((png_path, error))
    else:
        with multiprocessing.Pool(workers) as pool:
            for png_path, error in tqdm(
                pool.imap_unordered(convert_one, jobs, chunksize=8),
                total=len(jobs),
                desc=f"DICOM->PNG ({workers} procs)",
            ):
                if error:
                    failures.append((png_path, error))

    return {
        "converted": len(jobs) - len(failures),
        "skipped": skipped,
        "failures": failures,
    }
