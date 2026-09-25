import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from faceblast import (
    Face,
    MatchResult,
    compare_faces,
    detect_faces,
    draw_faces,
    get_face_embedding,
)
from faceblast.models import get_detector_model_path, get_recognizer_model_path


def test_models_downloaded_and_cached():
    det_path = Path(get_detector_model_path())
    rec_path = Path(get_recognizer_model_path())
    assert det_path.exists() and det_path.stat().st_size > 100_000
    assert rec_path.exists() and rec_path.stat().st_size > 10_000_000


def test_detect_blank_image():
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    faces = detect_faces(blank)
    assert faces == []


def test_compare_no_faces():
    blank1 = np.zeros((200, 200, 3), dtype=np.uint8)
    blank2 = np.zeros((200, 200, 3), dtype=np.uint8)
    result = compare_faces(blank1, blank2)
    assert isinstance(result, MatchResult)
    assert not result.is_match
    assert not result.face1_detected
    assert not result.face2_detected


def test_detect_and_compare_real_face():
    img_path1 = "tests/data/messi1.jpg"
    img_path2 = "tests/data/messi2.jpg"

    assert os.path.exists(img_path1)
    assert os.path.exists(img_path2)

    # Test detection
    faces = detect_faces(img_path1)
    assert len(faces) >= 1
    face = faces[0]
    assert isinstance(face, Face)
    assert len(face.landmarks) == 5
    assert face.confidence > 0.7

    # Test embedding extraction
    emb = get_face_embedding(img_path1, face=face)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[1] == 128  # SFace 128-dim embedding

    # Test comparison (same person)
    match_res = compare_faces(img_path1, img_path2)
    assert match_res.is_match
    assert match_res.similarity > 0.5


def test_draw_faces():
    img_path = "tests/data/messi1.jpg"
    faces = detect_faces(img_path)
    annotated = draw_faces(img_path, faces)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == cv2.imread(img_path).shape
