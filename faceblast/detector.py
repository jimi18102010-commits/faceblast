from dataclasses import dataclass
from typing import List, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from faceblast.models import get_detector_model_path


@dataclass
class Face:
    """Represents a detected face."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: List[Tuple[float, float]]  # 5 facial landmarks: eyes, nose, mouth corners
    raw_array: np.ndarray  # Raw 15-value vector from detector


def load_image(image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """Loads an image into a BGR numpy array format."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not load image from path: {image_input}")
        return img
    elif isinstance(image_input, Image.Image):
        rgb_img = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        return image_input
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")


class FaceDetector:
    """High-speed face detector powered by YuNet ONNX model."""

    def __init__(self, score_threshold: float = 0.6, nms_threshold: float = 0.3):
        self.model_path = get_detector_model_path()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000,
        )

    def detect(self, image_input: Union[str, np.ndarray, Image.Image]) -> List[Face]:
        """
        Detects faces in an image.

        Args:
            image_input: File path, numpy array, or PIL Image.

        Returns:
            List of detected Face objects.
        """
        img = load_image(image_input)
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return []

        results = []
        for row in faces:
            x, y, width, height = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            confidence = float(row[14])
            landmarks = [
                (float(row[4]), float(row[5])),    # right eye
                (float(row[6]), float(row[7])),    # left eye
                (float(row[8]), float(row[9])),    # nose tip
                (float(row[10]), float(row[11])),  # right mouth corner
                (float(row[12]), float(row[13])),  # left mouth corner
            ]
            results.append(
                Face(
                    bbox=(x, y, width, height),
                    confidence=confidence,
                    landmarks=landmarks,
                    raw_array=row,
                )
            )

        return results


# Global singleton instance for convenient functional usage
_default_detector: Union[FaceDetector, None] = None


def detect_faces(image_input: Union[str, np.ndarray, Image.Image], score_threshold: float = 0.6) -> List[Face]:
    """Detects faces in an image using default high-speed YuNet detector."""
    global _default_detector
    if _default_detector is None or _default_detector.score_threshold != score_threshold:
        _default_detector = FaceDetector(score_threshold=score_threshold)
    return _default_detector.detect(image_input)
