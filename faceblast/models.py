import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn

CACHE_DIR = Path.home() / ".cache" / "faceblast" / "models"
# Reviewed OpenCV Zoo commit. URLs must not track the mutable main branch.
MODEL_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
BASE_URL = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_COMMIT}/models"
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
YUNET_URL = f"{BASE_URL}/face_detection_yunet/{YUNET_FILENAME}"
SFACE_URL = f"{BASE_URL}/face_recognition_sface/{SFACE_FILENAME}"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file_with_progress(url: str, dest_path: Path, expected_sha256: str,
                                description: str = "Downloading"):
    """Download into a unique temp file; verify before atomically publishing it."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=dest_path.name + ".", suffix=".tmp", dir=dest_path.parent)
    temp_path = Path(temp_name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FaceBlast/0.1.0"})
        with os.fdopen(fd, "wb") as handle:
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get("content-length", 0))
                with Progress(SpinnerColumn("dots"), TextColumn(description), BarColumn(),
                              DownloadColumn(), TransferSpeedColumn(), TimeRemainingColumn()) as progress:
                    task = progress.add_task("download", total=total_size or None)
                    for chunk in iter(lambda: response.read(64 * 1024), b""):
                        handle.write(chunk)
                        progress.update(task, advance=len(chunk))
            handle.flush()
            os.fsync(handle.fileno())
        if file_sha256(temp_path) != expected_sha256:
            raise ValueError("Model SHA-256 verification failed; the untrusted download was discarded.")
        temp_path.replace(dest_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _model_path(filename: str, url: str, checksum: str, description: str) -> str:
    dest = CACHE_DIR / filename
    if not dest.is_file() or file_sha256(dest) != checksum:
        download_file_with_progress(url, dest, checksum, description)
    return str(dest)


def get_detector_model_path() -> str:
    return _model_path(YUNET_FILENAME, YUNET_URL, YUNET_SHA256, "YuNet Face Detector")


def get_recognizer_model_path() -> str:
    return _model_path(SFACE_FILENAME, SFACE_URL, SFACE_SHA256, "SFace Recognizer")
