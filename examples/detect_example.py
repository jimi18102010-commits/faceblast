"""
Example: Detecting faces and drawing bounding boxes using FaceBlast
"""

from faceblast import detect_faces, draw_faces

image_path = "tests/data/messi1.jpg"

# Detect all faces in image
faces = detect_faces(image_path)
print(f"Found {len(faces)} face(s)")

for i, face in enumerate(faces, 1):
    print(f"Face #{i}: bbox={face.bbox}, confidence={face.confidence:.1%}")
    print(f"  Landmarks (eyes, nose, mouth): {face.landmarks}")

# Draw and save visual result
draw_faces(image_path, faces, output_path="faces_annotated.jpg")
print("Saved annotated image to faces_annotated.jpg")
