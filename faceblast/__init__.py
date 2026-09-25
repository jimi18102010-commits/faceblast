"""
FaceBlast: Ultra-fast, lightweight face detection and recognition using ONNX.
Zero C++ CMake compilation (no dlib) and Zero PyTorch/TensorFlow bloat.
"""

from faceblast.detector import Face, FaceDetector, detect_faces, load_image
from faceblast.recognizer import (
    FaceRecognizer,
    MatchResult,
    compare_faces,
    get_face_embedding,
)
from faceblast.ui import draw_faces

__version__ = "0.1.0"
__all__ = [
    "Face",
    "FaceDetector",
    "detect_faces",
    "load_image",
    "FaceRecognizer",
    "MatchResult",
    "compare_faces",
    "get_face_embedding",
    "draw_faces",
    "__version__",
]
