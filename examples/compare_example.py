"""
Example: Comparing two face images using FaceBlast
"""

from faceblast import compare_faces

# Compare two photos
result = compare_faces("tests/data/messi1.jpg", "tests/data/messi2.jpg")

print(f"Is same person: {result.is_match}")
print(f"Similarity score: {result.similarity:.2%}")
print(f"L2 Distance: {result.distance_l2:.3f}")
