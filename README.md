# ⚡ FaceBlast

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-jimi18102010--commits-181717?logo=github)](https://github.com/jimi18102010-commits/faceblast)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI Tests](https://github.com/jimi18102010-commits/faceblast/actions/workflows/test.yml/badge.svg)](https://github.com/jimi18102010-commits/faceblast/actions/workflows/test.yml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-orange.svg)]()

**Ultra-fast, lightweight face detection and recognition using ONNX.**  
*Zero C++ CMake compilation (no `dlib`) and Zero PyTorch/TensorFlow bloat.*

[📖 Русская версия руководства (Russian Guide)](docs/GUIDE_RU.md)

</div>

---

```text
╭─ Comparison: person1.jpg ↔ person2.jpg ─╮
│ ✔ SAME PERSON (MATCH)                   │
│                                         │
│ • Similarity: 0.7970 (79.7%)            │
│ • Threshold: ≥ 0.3630                   │
│ • L2 Distance: 0.637                    │
╰─────────────────────────────────────────╯
```

---

## Implementation and limits

FaceBlast uses OpenCV's YuNet and SFace APIs to load ONNX model files.
It does **not** use the separate ONNX Runtime package, dlib, PyTorch or TensorFlow.
Models are pinned to an OpenCV Zoo commit and verified with SHA-256, including cached copies.
The first use downloads model weights; latency and RAM depend on hardware and image size.
No universal performance benchmark is claimed.

Similarity is not identity proof or a calibrated probability. Obtain consent for biometric
processing and do not use this package alone for high-stakes authentication or identification.
The functional API caches detector/recognizer instances; use separate processes or external
serialization when sharing them between concurrent workers.

## 📦 Installation

```bash
git clone https://github.com/jimi18102010-commits/faceblast.git
cd faceblast

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Bash / Zsh
# OR for Fish Shell (CachyOS / Arch):
# source .venv/bin/activate.fish

# Install editable package
pip install -e .
```

---

## 💻 CLI Usage

### 1. Interactive Mode
Run without arguments for an interactive guide:
```bash
faceblast
```

### 2. Compare Two Faces (Verification)
Check if two photos belong to the same individual:
```bash
faceblast compare photo1.jpg photo2.jpg
```

### 3. Detect Faces & Landmarks
Detect all faces in an image and draw bounding boxes:
```bash
faceblast detect crowd.jpg --draw -o annotated.jpg
```

---

## 🐍 Python API Usage

### Compare Two Faces
```python
from faceblast import compare_faces

result = compare_faces("person1.jpg", "person2.jpg")

if result.is_match:
    print(f"Match found! Similarity: {result.similarity:.1%}")
else:
    print("Different people.")
```

### Detect Faces & Extract 5 Landmarks
```python
from faceblast import detect_faces, draw_faces

# Detect faces
faces = detect_faces("group_photo.jpg")

for face in faces:
    x, y, w, h = face.bbox
    print(f"Face Box: ({x}, {y}, {w}, {h}) | Confidence: {face.confidence:.1%}")
    print(f"Landmarks (eyes, nose, mouth): {face.landmarks}")

# Annotate image
draw_faces("group_photo.jpg", faces, output_path="detected.jpg")
```

### Extract 128-D Embeddings
```python
from faceblast import get_face_embedding

# Returns a 128-dimensional numpy embedding vector
embedding = get_face_embedding("face.jpg")
print(embedding.shape)  # (1, 128)
```

---

## 🧪 Testing

Run automated tests:

```bash
pytest
```

---

## 👤 Author

Developed by **[jimmiy](https://github.com/jimi18102010-commits)**.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
