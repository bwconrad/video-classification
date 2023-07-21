from pytorchvideo.models import create_resnet
from torch import nn


def create_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    if name == "resnet18-3d":
        return create_resnet(
            model_depth=18, input_channel=3, model_num_class=num_classes
        )
    elif name == "resnet50-3d":
        return create_resnet(
            model_depth=50, input_channel=3, model_num_class=num_classes
        )
    elif name == "resnet101-3d":
        return create_resnet(
            model_depth=101, input_channel=3, model_num_class=num_classes
        )
    else:
        raise ValueError(f"{name} is not an available model.")
