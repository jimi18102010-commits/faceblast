from typing import List, Optional, Union
import cv2
import numpy as np
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from faceblast.detector import Face, load_image
from faceblast.recognizer import MatchResult

console = Console()


def print_banner():
    """Prints FaceBlast CLI banner."""
    console.print(
        Panel.fit(
            "[bold cyan]⚡ FaceBlast[/bold cyan] [dim]v0.1.0[/dim]\n"
            "[italic dim]Ultra-Fast Face Detection & Recognition on ONNX (Zero C++ / Zero PyTorch bloat)[/italic dim]",
            border_style="cyan"
        )
    )


def print_detection_results(faces: List[Face], image_path: str):
    """Prints detection stats in a clean Rich table."""
    if not faces:
        console.print(f"[yellow]⚠ No faces detected in: {image_path}[/yellow]")
        return

    table = Table(title=f"Detected {len(faces)} Face(s) in {image_path}", border_style="cyan")
    table.add_column("Face #", justify="center", style="bold")
    table.add_column("Box (X, Y, W, H)", justify="center")
    table.add_column("Confidence", justify="center", style="green")
    table.add_column("Eyes & Nose Coordinates", justify="left")

    for i, face in enumerate(faces, 1):
        x, y, w, h = face.bbox
        r_eye, l_eye, nose, r_mouth, l_mouth = face.landmarks
        table.add_row(
            str(i),
            f"{x}, {y}, {w}x{h}",
            f"{face.confidence:.1%}",
            f"Eyes: ({int(r_eye[0])},{int(r_eye[1])}), ({int(l_eye[0])},{int(l_eye[1])}) | Nose: ({int(nose[0])},{int(nose[1])})"
        )

    console.print(table)


def print_match_result(result: MatchResult, img1_name: str, img2_name: str):
    """Prints a styled panel comparing two faces."""
    if not result.face1_detected:
        console.print(f"[bold red]✘ Could not detect any face in:[/bold red] {img1_name}")
        return
    if not result.face2_detected:
        console.print(f"[bold red]✘ Could not detect any face in:[/bold red] {img2_name}")
        return

    score_pct = max(0.0, min(100.0, result.similarity * 100))
    if result.is_match:
        content = (
            f"[bold green]✔ SAME PERSON (MATCH)[/bold green]\n\n"
            f"• [bold]Similarity:[/bold] {result.similarity:.4f} ({score_pct:.1f}%)\n"
            f"• [bold]Threshold:[/bold] ≥ 0.3630\n"
            f"• [bold]L2 Distance:[/bold] {result.distance_l2:.3f}"
        )
        border = "green"
    else:
        content = (
            f"[bold red]✘ DIFFERENT PERSONS (NO MATCH)[/bold red]\n\n"
            f"• [bold]Similarity:[/bold] {result.similarity:.4f} ({score_pct:.1f}%)\n"
            f"• [bold]Threshold:[/bold] < 0.3630\n"
            f"• [bold]L2 Distance:[/bold] {result.distance_l2:.3f}"
        )
        border = "red"

    console.print(
        Panel(
            content,
            title=f"Comparison: {img1_name} ↔ {img2_name}",
            border_style=border,
            expand=False
        )
    )


def draw_faces(
    image_input: Union[str, np.ndarray, Image.Image],
    faces: List[Face],
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Draws bounding boxes, landmark points, and confidence labels on image.

    Args:
        image_input: Path or array of the image.
        faces: List of detected Face objects.
        output_path: Optional file path to save the drawn image.

    Returns:
        Annotated BGR numpy image.
    """
    img = load_image(image_input).copy()

    for i, face in enumerate(faces, 1):
        x, y, w, h = face.bbox

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 128), 2)

        # Draw label header
        label = f"Face #{i} ({face.confidence:.0%})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, max(0, y - 22)), (x + label_size[0] + 8, y), (0, 255, 128), -1)
        cv2.putText(img, label, (x + 4, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw 5 landmarks
        colors = [
            (255, 0, 0),    # right eye (blue)
            (0, 0, 255),    # left eye (red)
            (0, 255, 255),  # nose (yellow)
            (255, 0, 255),  # right mouth (magenta)
            (0, 255, 0),    # left mouth (green)
        ]
        for (lx, ly), color in zip(face.landmarks, colors):
            cv2.circle(img, (int(lx), int(ly)), 4, color, -1, cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, img)

    return img
