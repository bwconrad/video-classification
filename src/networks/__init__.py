from torch import nn
from torchvision.models.video import r3d_18


def create_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    if name == "r3d-18":
        return r3d_18(num_classes=num_classes)
    else:
        raise ValueError(f"{name} is not an available model.")
