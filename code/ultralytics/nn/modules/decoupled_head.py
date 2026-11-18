# import copy
# import math

# import torch
# import torch.nn as nn
# from torch.nn.init import constant_, xavier_uniform_

# from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

# from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
# from .conv import Conv
# from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
# from .utils import bias_init_with_prob, linear_init
# __all__ = "DetectLesion", "DetectFish"


# class _DecoupledDetect(nn.Module):
#     """Decoupled head for detection models."""
#     dynamic = False  # force grid reconstruction
#     export = False  # export mode
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init

#     def __init__(self, nc: int, ch: tuple):
#         """Initializes the decoupled head with specified number of classes and channels."""
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16  # DFL channels (ch[0] //
#         self.no = nc + self.reg_max * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build
        
#         c2 = max((16, ch[0] // 4, self.reg_max * 4))  # channels
#         c3 = max(ch[0], min(self.nc, 100))  # channels

#         self.cv2 = nn.ModuleList(
#             nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
#         )
#         self.cv3 = nn.ModuleList(
#             nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
#         )
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

#     def forward(self, x):
#         """Concatenates and returns predicted bounding boxes and class probabilities."""
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         if self.training:  # Training path
#             return x
        
#         shape = x[0].shape  # BCHW
#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
#         if self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape
        
#         box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
#         dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
#         y = torch.cat((dbox, cls.sigmoid()), 1)
#         return y if self.export else (y, x)
    
#     def bias_init(self):
#         """Initialize Detect() biases, WARNING: requires stride availability."""
#         m = self  # self.model[-1]  # Detect() module
#         for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
#             a[-1].bias.data[:] = 1.0  # box
#             b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
    

# class DetectLesion(_DecoupledDetect):
#     """
#     lesion only
#     只接受P3特征图，只输出病斑检测
#     """

#     head_name = "lesion"

#     def __init__(self, nc=4, ch=()):
#         """Initializes the DetectLesion object with the specified number of classes and input channels."""
#         super().__init__(nc=nc, ch=tuple(ch) if not isinstance(ch, tuple) else ch)


# class DetectFish(_DecoupledDetect):
#     """
#     fish only
#     只接受P4、P5特征图，只输出鱼体检测
#     """

#     head_name = "fish"

#     def __init__(self, nc=1, ch=()):
#         """Initializes the DetectionFish object with the specified number of classes and input channels."""
#         super().__init__(nc=nc, ch=tuple(ch) if not isinstance(ch, tuple) else ch)


"""Robust task-decoupled detection heads for Ultralytics YOLOv8.

Target environment
------------------
Ultralytics 8.2.42 (or another release exposing ``Detect`` from
``ultralytics.nn.modules.head`` with the same constructor and forward API).

Global dataset class encoding
-----------------------------
0: Fish Body
1: Viral
2: Bacterial
3: Fungal
4: Mechanical injury

The fish head uses one local class (local 0 -> global 0).  The lesion head
uses four local classes (local 0..3 -> global 1..4).  Keeping this mapping on
the head modules allows the loss and inference-merging code to validate the
same class convention instead of maintaining independent hard-coded lists.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
from torch import Tensor

from .head import Detect

__all__ = (
    "DetectLesion",
    "DetectFish",
    "merge_decoupled_predictions",
)


def _validate_channels(ch: Sequence[int], head_name: str) -> Tuple[int, ...]:
    """Return a validated immutable channel specification."""
    if ch is None:
        raise ValueError("{} requires a non-empty channel list.".format(head_name))

    channels = tuple(int(value) for value in ch)
    if not channels:
        raise ValueError("{} requires at least one input feature map.".format(head_name))
    if any(value <= 0 for value in channels):
        raise ValueError(
            "{} received invalid input channels: {}.".format(head_name, channels)
        )
    return channels


class _DecoupledDetect(Detect):
    """Base class that adds an explicit local-to-global class mapping."""

    head_name = "decoupled"

    def __init__(
        self,
        nc: int,
        ch: Sequence[int],
        global_class_ids: Sequence[int],
    ) -> None:
        nc = int(nc)
        if nc <= 0:
            raise ValueError("{} requires nc > 0, received {}.".format(self.head_name, nc))

        channels = _validate_channels(ch, self.head_name)
        global_ids = tuple(int(value) for value in global_class_ids)

        if len(global_ids) != nc:
            raise ValueError(
                "{} has nc={} but {} global class IDs were supplied: {}.".format(
                    self.head_name, nc, len(global_ids), global_ids
                )
            )
        if any(value < 0 for value in global_ids):
            raise ValueError(
                "{} global class IDs must be non-negative: {}.".format(
                    self.head_name, global_ids
                )
            )
        if len(set(global_ids)) != len(global_ids):
            raise ValueError(
                "{} global class IDs must be unique: {}.".format(
                    self.head_name, global_ids
                )
            )

        super().__init__(nc=nc, ch=channels)

        # Tuple form is convenient for configuration checks.  The buffer form
        # follows the module device and is used during inference remapping.
        self.global_class_ids = global_ids
        self.register_buffer(
            "_global_class_ids_tensor",
            torch.tensor(global_ids, dtype=torch.long),
            persistent=False,
        )

    def expand_predictions(self, prediction: Tensor, total_nc: int) -> Tensor:
        """Expand local class scores to the global dataset class space.

        Args:
            prediction: Decoded prediction tensor with shape
                ``[batch, 4 + self.nc, anchors]``.
            total_nc: Number of classes in the global dataset.

        Returns:
            Tensor with shape ``[batch, 4 + total_nc, anchors]``.  Classes not
            handled by this head receive zero scores.
        """
        if not torch.is_tensor(prediction) or prediction.ndim != 3:
            raise ValueError(
                "{} prediction must be a 3-D tensor [B, 4+nc, N].".format(
                    self.head_name
                )
            )

        expected_channels = 4 + self.nc
        if prediction.shape[1] != expected_channels:
            raise ValueError(
                "{} prediction has {} channels; expected {} (= 4 + nc).".format(
                    self.head_name, prediction.shape[1], expected_channels
                )
            )

        total_nc = int(total_nc)
        minimum_nc = max(self.global_class_ids) + 1
        if total_nc < minimum_nc:
            raise ValueError(
                "total_nc={} cannot represent {} mapping {}.".format(
                    total_nc, self.head_name, self.global_class_ids
                )
            )

        expanded = prediction.new_zeros(
            prediction.shape[0], 4 + total_nc, prediction.shape[2]
        )
        expanded[:, :4, :] = prediction[:, :4, :]
        class_indices = self._global_class_ids_tensor.to(prediction.device) + 4
        expanded[:, class_indices, :] = prediction[:, 4:, :]
        return expanded


class DetectLesion(_DecoupledDetect):
    """P3 lesion head: local classes 0..3 map to global classes 1..4."""

    head_name = "lesion"

    def __init__(self, nc: int = 3, ch: Sequence[int] = ()) -> None:
        nc = int(nc)
        # Fish Body occupies global class 0.  Lesion classes therefore begin
        # at global class 1 and preserve the order in dataset.yaml.
        super().__init__(nc=nc, ch=ch, global_class_ids=tuple(range(1, nc + 1)))


class DetectFish(_DecoupledDetect):
    """P4/P5 fish head: local class 0 maps to global class 0."""

    head_name = "fish"

    def __init__(self, nc: int = 1, ch: Sequence[int] = ()) -> None:
        nc = int(nc)
        if nc != 1:
            raise ValueError(
                "DetectFish is defined for the single Fish Body class; "
                "expected nc=1, received nc={}.".format(nc)
            )
        super().__init__(nc=nc, ch=ch, global_class_ids=(0,))


DecodedOutput = Union[Tensor, Tuple[Tensor, object]]


def _decoded_tensor(output: DecodedOutput, head_name: str) -> Tensor:
    """Extract the decoded tensor from a Detect inference return value."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and len(output) >= 1 and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(
        "{} inference output must be a tensor or (decoded_tensor, raw_features).".format(
            head_name
        )
    )


def merge_decoupled_predictions(
    lesion_output: DecodedOutput,
    fish_output: DecodedOutput,
    lesion_head: DetectLesion,
    fish_head: DetectFish,
    total_nc: int = 4,
) -> Tensor:
    """Map both heads to global classes and concatenate their predictions.

    This helper should be used by the model inference path before standard
    Ultralytics post-processing/NMS.  Concatenating the local outputs directly
    would incorrectly interpret lesion-local class 0 as global class 0.
    """
    if not isinstance(lesion_head, DetectLesion):
        raise TypeError("lesion_head must be a DetectLesion instance.")
    if not isinstance(fish_head, DetectFish):
        raise TypeError("fish_head must be a DetectFish instance.")

    lesion_prediction = _decoded_tensor(lesion_output, "lesion")
    fish_prediction = _decoded_tensor(fish_output, "fish")

    if lesion_prediction.shape[0] != fish_prediction.shape[0]:
        raise ValueError(
            "Lesion and fish predictions have different batch sizes: {} and {}.".format(
                lesion_prediction.shape[0], fish_prediction.shape[0]
            )
        )

    lesion_global = lesion_head.expand_predictions(lesion_prediction, total_nc)
    fish_global = fish_head.expand_predictions(fish_prediction, total_nc)
    return torch.cat((lesion_global, fish_global), dim=2)

