from pathlib import Path

from PIL import Image

from .inference import CIFAR10Inference


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def main():
    predictor = CIFAR10Inference()

    images = sorted(
        path
        for path in VALIDATION_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No images found in {VALIDATION_DIR}")
        return

    for image_path in images:
        with Image.open(image_path) as image:
            result = predictor.predict(image)

        print(
            f"{image_path.name:30}"
            f"{result['class']:10}"
            f"{result['confidence'] * 100:6.2f}%"
        )


if __name__ == "__main__":
    main()
