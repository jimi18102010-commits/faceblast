from dataclasses import dataclass
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from faceblast.detector import Face, detect_faces, load_image
from faceblast.models import get_recognizer_model_path

# Official SFace cosine similarity matching threshold
DEFAULT_COSINE_THRESHOLD = 0.363


@dataclass
class MatchResult:
    """Result of comparing two faces."""
    is_match: bool
    similarity: float
    distance_l2: float
    face1_detected: bool
    face2_detected: bool


class FaceRecognizer:
    """Face recognition and embedding extraction powered by SFace ONNX."""

    def __init__(self, cosine_threshold: float = DEFAULT_COSINE_THRESHOLD):
        self.model_path = get_recognizer_model_path()
        self.cosine_threshold = cosine_threshold
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config="",
        )

    def extract_embedding(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        face: Optional[Face] = None
    ) -> Tuple[np.ndarray, Optional[Face]]:
        """
        Extracts a 128-dimensional feature embedding for the face in the image.

        Args:
            image_input: File path, numpy array, or PIL Image.
            face: Optional pre-detected Face object. If None, detects automatically.

        Returns:
            (embedding, Face)
        """
        img = load_image(image_input)

        if face is None:
            detected_faces = detect_faces(img)
            if not detected_faces:
                raise ValueError("No face detected in the provided image.")
            face = detected_faces[0]

        aligned_face = self.recognizer.alignCrop(img, face.raw_array)
        embedding = self.recognizer.feature(aligned_face)
        return embedding, face

    def compare(
        self,
        image1: Union[str, np.ndarray, Image.Image],
        image2: Union[str, np.ndarray, Image.Image],
        threshold: Optional[float] = None
    ) -> MatchResult:
        """
        Compares two face images to determine if they are the same person.

        Args:
            image1: First face image path or array.
            image2: Second face image path or array.
            threshold: Cosine similarity threshold (default: 0.363).

        Returns:
            MatchResult with is_match, similarity score, and distance.
        """
        thresh = threshold if threshold is not None else self.cosine_threshold

        try:
            feat1, face1 = self.extract_embedding(image1)
            face1_found = True
        except ValueError:
            return MatchResult(False, 0.0, 999.0, False, False)

        try:
            feat2, face2 = self.extract_embedding(image2)
            face2_found = True
        except ValueError:
            return MatchResult(False, 0.0, 999.0, True, False)

        # Match using cosine similarity and L2 norm
        cosine_sim = float(self.recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE))
        l2_dist = float(self.recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_NORM_L2))

        is_match = cosine_sim >= thresh

        return MatchResult(
            is_match=is_match,
            similarity=cosine_sim,
            distance_l2=l2_dist,
            face1_detected=face1_found,
            face2_detected=face2_found,
        )


_default_recognizer: Optional[FaceRecognizer] = None


def compare_faces(
    image1: Union[str, np.ndarray, Image.Image],
    image2: Union[str, np.ndarray, Image.Image],
    threshold: float = DEFAULT_COSINE_THRESHOLD
) -> MatchResult:
    """Compares two images and returns whether they contain the same person."""
    global _default_recognizer
    if _default_recognizer is None:
        _default_recognizer = FaceRecognizer(cosine_threshold=threshold)
    return _default_recognizer.compare(image1, image2, threshold=threshold)


def get_face_embedding(
    image_input: Union[str, np.ndarray, Image.Image],
    face: Optional[Face] = None
) -> np.ndarray:
    """Extracts a 128-dimensional embedding vector from a face."""
    global _default_recognizer
    if _default_recognizer is None:
        _default_recognizer = FaceRecognizer()
    embedding, _ = _default_recognizer.extract_embedding(image_input, face=face)
    return embedding
