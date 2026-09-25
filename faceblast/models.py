import os
import urllib.request
from pathlib import Path
from typing import Optional
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

CACHE_DIR = Path.home() / ".cache" / "faceblast" / "models"

YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"


def download_file_with_progress(url: str, dest_path: Path, description: str = "Downloading"):
    """Downloads a file from URL to dest_path with a Rich progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    req = urllib.request.Request(url, headers={"User-Agent": "FaceBlast/0.1.0"})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("content-length", 0))

        with Progress(
            SpinnerColumn("dots", style="cyan"),
            TextColumn(f"[bold cyan]{description}"),
            BarColumn(bar_width=35, complete_style="cyan", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("download", total=total_size)
            chunk_size = 64 * 1024  # 64 KB

            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    temp_path.replace(dest_path)


def get_detector_model_path() -> str:
    """Returns local path to YuNet face detection ONNX model, downloading if necessary."""
    dest = CACHE_DIR / YUNET_FILENAME
    if not dest.exists() or dest.stat().st_size == 0:
        download_file_with_progress(YUNET_URL, dest, description="YuNet Face Detector (0.3 MB)")
    return str(dest)


def get_recognizer_model_path() -> str:
    """Returns local path to SFace recognition ONNX model, downloading if necessary."""
    dest = CACHE_DIR / SFACE_FILENAME
    if not dest.exists() or dest.stat().st_size == 0:
        download_file_with_progress(SFACE_URL, dest, description="SFace Recognizer (37 MB)")
    return str(dest)
