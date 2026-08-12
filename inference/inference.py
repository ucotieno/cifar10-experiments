from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .model import load_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "cifar10_net.pth"


class CIFAR10Inference:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model, artifact = load_model(model_path)

        self.classes = tuple(artifact["classes"])

        input_width, input_height = artifact["input_size"]

        normalize_mean = tuple(artifact["normalize_mean"])
        normalize_std = tuple(artifact["normalize_std"])

        self.transform = transforms.Compose([
            transforms.Resize((input_height, input_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                normalize_mean,
                normalize_std,
            ),
        ])

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)

            confidence, predicted = torch.max(
                probabilities,
                dim=1,
            )

        class_index = predicted.item()

        return {
            "class": self.classes[class_index],
            "class_index": class_index,
            "confidence": float(confidence.item()),
        }
