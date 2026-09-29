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

## 🚀 Why FaceBlast?

For years, developers have struggled with two extremes in Python face recognition:
1. **`face_recognition` (53k ⭐):** Relies on `dlib`, which requires `CMake` and a C++ compiler to install. Fails frequently on Linux, macOS (Apple Silicon M1/M2/M3), and inside minimal Docker containers.
2. **`DeepFace` (~15k ⭐):** Pulls in `TensorFlow` or `PyTorch`, causing **2+ GB** package downloads, 10-second cold starts, and 1.5 GB memory footprints.

**FaceBlast strikes the sweet spot:**

| Feature | `face_recognition` (dlib) | `DeepFace` | **FaceBlast** ⚡ |
| :--- | :---: | :---: | :---: |
| **Installation** | ❌ Requires C++ & CMake | ⚠️ 2+ GB package bloat | ✅ **Pure wheel (`pip install`)** |
| **Under the Hood** | Outdated `dlib` (2020) | Heavy PyTorch / TF | ✅ **Fast ONNX (YuNet + SFace)** |
| **Startup Time** | ~400 ms | 8,000 - 15,000 ms | ✅ **~50 ms (Instant)** |
| **RAM Footprint** | ~200 MB | ~1,500 MB | ✅ **< 100 MB** |
| **Facial Landmarks** | 68 (Slow) | Varies | ✅ **5 key points (Instant)** |
| **Interactive CLI** | Basic | No | ✅ **Rich terminal UI & tables** |

---

## 📦 Installation

```bash
git clone https://github.com/jimi18102010-commits/faceblast.git
cd faceblast

# Setup virtual environment
uv venv .venv --python 3.12
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
