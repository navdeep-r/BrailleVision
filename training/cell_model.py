"""
training/cell_model.py

MobileNetV3Small CNN classifier for Braille dot pattern recognition.

65 classes:
  0-63  = 6-bit dot pattern integer (bit 0 = dot1 top-left … bit 5 = dot6 bot-right)
  64    = blank/space sentinel

WHY MobileNetV3Small:
  - Small enough to run on CPU during live demo
  - ImageNet pretraining gives strong edge/texture feature extraction
    that transfers well to Braille dot pattern discrimination
"""

import torch
import torch.nn as nn
from torchvision import models


class BrailleCellClassifier(nn.Module):
    """
    MobileNetV3Small with a replaced final head for Braille cell classification.
    """

    NUM_CLASSES = 65  # 0-63 bit patterns + 64 blank

    def __init__(self, num_classes: int = 65):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

        # Replace the final linear layer
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, num_classes)

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @classmethod
    def load_for_inference(
        cls,
        weights_path: str,
        device: str = "cpu",
    ) -> "BrailleCellClassifier":
        """
        Load a saved checkpoint for inference.
        Sets model to eval mode and moves to the requested device.
        """
        model = cls()
        ckpt = torch.load(weights_path, map_location=device)

        # Support both raw state_dict and wrapped checkpoint formats
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
