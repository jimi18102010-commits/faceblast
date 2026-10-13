# 📘 Руководство пользователя FaceBlast (на русском языке)

**FaceBlast** — это сверхбыстрая и легковесная Python-библиотека и CLI-утилита для распознавания и обнаружения лиц на базе ONNX (YuNet + SFace).

Это экспериментальный инструмент; сходство не является доказательством личности. Используйте изображения с согласия людей.

---

## 🚀 Почему FaceBlast?

* **Никаких компиляторов C++ (CMake):** Устанавливается через обычный `pip install` за пару секунд.
* **Никакого тяжелого PyTorch/TensorFlow:** Библиотека не выкачивает 2 ГБ лишних пакетов.
* Модели ONNX загружаются через OpenCV, закреплены за commit и проверяются SHA-256.
* Скорость и RAM зависят от оборудования и изображения; универсального бенчмарка нет.

---

## ⚡ Быстрый старт

### 1. Установка

```bash
git clone https://github.com/jimi18102010-commits/faceblast.git
cd faceblast
python3 -m venv .venv
source .venv/bin/activate       # Bash/Zsh в Ubuntu/Debian
pip install -e .
```

---

## 🎮 Использование в терминале (CLI)

### Интерактивный режим
Просто напишите:
```bash
faceblast
```
Утилита покажет меню: сравнить два фото или найти лица на изображении.

### Сравнить два лица (один человек или разные?)
```bash
faceblast compare person1.jpg person2.jpg
```
Результат покажет процент сходства (Similarity) и вердикт: `SAME PERSON` или `DIFFERENT PERSONS`.

### Найти лица на фото и нарисовать рамки
```bash
faceblast detect photo.jpg --draw -o result.jpg
```
Программа найдет все лица, определит координаты глаз, носа и углов рта и сохранит изображение `result.jpg`.

---

## 🐍 Использование в Python

### 1. Сравнение двух лиц
```python
from faceblast import compare_faces

result = compare_faces("photo1.jpg", "photo2.jpg")

if result.is_match:
    print(f"Это один человек! Сходство: {result.similarity:.1%}")
else:
    print("Это разные люди.")
```

### 2. Поиск лиц и ключевых точек
```python
from faceblast import detect_faces, draw_faces

# Найти все лица
faces = detect_faces("crowd.jpg")

for face in faces:
    x, y, w, h = face.bbox
    print(f"Лицо: {w}x{h}, уверенность: {face.confidence:.1%}")
    print(f"Глаза, нос, рот: {face.landmarks}")

# Отрисовать рамки на фото
draw_faces("crowd.jpg", faces, output_path="output.jpg")
```
