import argparse
import os
import sys
from types import SimpleNamespace
from rich.prompt import Prompt

from faceblast import __version__
from faceblast.detector import detect_faces
from faceblast.recognizer import compare_faces
from faceblast.ui import (
    console,
    draw_faces,
    print_banner,
    print_detection_results,
    print_match_result,
)


def handle_detect(args):
    """Handles face detection on a single image."""
    image_path = args.image
    if not os.path.exists(image_path):
        console.print(f"[bold red]✘ File not found:[/bold red] {image_path}")
        sys.exit(1)

    with console.status("[bold cyan]Scanning for faces..."):
        faces = detect_faces(image_path, score_threshold=args.threshold)

    print_detection_results(faces, image_path)

    if args.output or args.draw:
        output_file = args.output or f"detected_{os.path.basename(image_path)}"
        draw_faces(image_path, faces, output_path=output_file)
        console.print(f"[bold green]✔ Annotated image saved to:[/bold green] {output_file}")


def handle_compare(args):
    """Handles comparing two face images."""
    if not os.path.exists(args.image1):
        console.print(f"[bold red]✘ File not found:[/bold red] {args.image1}")
        sys.exit(1)
    if not os.path.exists(args.image2):
        console.print(f"[bold red]✘ File not found:[/bold red] {args.image2}")
        sys.exit(1)

    with console.status("[bold cyan]Analyzing & comparing faces..."):
        result = compare_faces(args.image1, args.image2, threshold=args.threshold)

    print_match_result(result, os.path.basename(args.image1), os.path.basename(args.image2), threshold=args.threshold)


def handle_interactive():
    """Interactive mode when no subcommands are passed."""
    print_banner()
    console.print("[cyan]Choose an action:[/cyan]")
    console.print("1. [bold]Compare two faces[/bold] (verify if it is the same person)")
    console.print("2. [bold]Detect faces in an image[/bold] (draw bounding boxes and landmarks)")
    console.print("3. [bold]Exit[/bold]\n")

    choice = Prompt.ask("Select option", choices=["1", "2", "3"], default="1")
    if choice == "1":
        img1 = Prompt.ask("Path to Photo 1").strip(" '\"")
        img2 = Prompt.ask("Path to Photo 2").strip(" '\"")
        handle_compare(SimpleNamespace(image1=img1, image2=img2, threshold=0.363))
    elif choice == "2":
        img = Prompt.ask("Path to Photo").strip(" '\"")
        draw = Prompt.ask("Save annotated image with boxes?", choices=["y", "n"], default="y")
        handle_detect(SimpleNamespace(image=img, threshold=0.6, output=None, draw=(draw == "y")))


def main():
    parser = argparse.ArgumentParser(
        prog="faceblast",
        description="⚡ FaceBlast: Ultra-fast face detection & recognition powered by ONNX",
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: detect
    detect_parser = subparsers.add_parser("detect", help="Detect faces and facial landmarks in an image")
    detect_parser.add_argument("image", type=str, help="Path to input image")
    detect_parser.add_argument("-o", "--output", type=str, default=None, help="Save annotated image to path")
    detect_parser.add_argument("--draw", action="store_true", help="Draw bounding boxes and landmarks")
    detect_parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold (default: 0.6)")

    # Command: compare
    compare_parser = subparsers.add_parser("compare", help="Compare two face photos to check if they match")
    compare_parser.add_argument("image1", type=str, help="Path to first photo")
    compare_parser.add_argument("image2", type=str, help="Path to second photo")
    compare_parser.add_argument("--threshold", type=float, default=0.363, help="Cosine threshold (default: 0.363)")

    args = parser.parse_args()

    if not args.command:
        handle_interactive()
    else:
        print_banner()
        if args.command == "detect":
            handle_detect(args)
        elif args.command == "compare":
            handle_compare(args)


if __name__ == "__main__":
    main()
