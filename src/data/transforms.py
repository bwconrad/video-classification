from typing import Any

import torch
from torchvision.transforms.v2 import InterpolationMode, Transform
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional._geometry import _check_interpolation
from torchvision.transforms.v2.utils import query_spatial_size


class RandomShortSideScale(Transform):
    """
    Transformation that resizes the shortest side of an image
    randomly between a minimum and maximum size.

    Args:
        min_size (int): The minimum size for the shortest side.
        max_size (int): The maximum size for the shortest side.
        interpolation (InterpolationMode or int): The interpolation mode to use for resizing. Default is InterpolationMode.BILINEAR.
        antialias (str, bool, or None): Whether to apply antialiasing. Default is True.
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        interpolation: InterpolationMode | int = InterpolationMode.BILINEAR,
        antialias: str | bool | None = True,
    ) -> None:
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = _check_interpolation(interpolation)
        self.antialias = antialias

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """
        Randomly selects height and width resizing parameters based on the input image.

        Args:
            flat_inputs (list[Any]): The flattened input list.

        Returns:
            dict[str, Any]: The resizing parameters.
        """
        height, width = query_spatial_size(flat_inputs)
        short_side = min(height, width)
        target_size = torch.randint(self.min_size, self.max_size + 1, (1,)).item()
        aspect_ratio = float(short_side) / target_size

        if width < height:
            new_width = target_size
            new_height = int(height / aspect_ratio)
        else:
            new_width = int(width / aspect_ratio)
            new_height = target_size

        return dict(size=(new_height, new_width))

    def _transform(self, inp: Any, params: dict[str, Any]) -> Any:
        """
        Applies the resizing transformation to the input image.

        Args:
            inp (Any): The input image.
            params (dict[str, Any]): The resizing parameters.

        Returns:
            Any: The resized image.
        """
        return F.resize(
            inp,
            size=params["size"],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
