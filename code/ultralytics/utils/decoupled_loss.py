# import torch
# import torch.nn as nn
# from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
# from ultralytics.utils.metrics import bbox_iou
# from ultralytics.utils.ops import xywh2xyxy

# class _SingleHeadLoss:
#     """
#     单个解耦检测头的loss计算
#     与v8DetectionLoss类似，但是只处理指定类别的gt

#     args:
#         model_head: DetectLesion 或 DetectFish
#         class_ids: 当前头负责的原始类别id列表，例如 [3] 或 [0, 1, 2]

#         class_weights: 类别权重
#     """
#     def __init__(self, model_head, class_ids: list, class_weights: float = 0.5):
#         self.model_head = model_head
#         self.class_ids = class_ids
#         self.class_weights = class_weights
#         self.nc = model_head.nc  # 当前头的类别数
#         self.no = model_head.no  # 每个anchor的输出维度
#         self.reg_max = model_head.reg_max
#         # self.device = next(model_head.parameters()).device

#         self.bce = nn.BCEWithLogitsLoss(reduction='none')
#         self.assigner = TaskAlignedAssigner(topk=13,
#                                             num_classes=self.nc,
#                                             alpha=1.0,
#                                             beta=6.0)  # TAL分配器，topk可调
#         self.bbox_loss = _BBoxLoss(self.reg_max - 1)  # 负责边界框回归损失计算的辅助类

#         self.proj = torch.arange(self.reg_max, dtype=torch.float)  # 用于将分布式回归转换为坐标
    
#     def preprocess(self, targets, batch_size, scale_tensor):
#         """
#         过滤出本头负责的类别gt，并且把类别id重映射到本头的内部id。

#         returns:
#             targets:(n, 6) [batch_idx, local_cls_id, x_center, y_center, width, height]  只包含本头负责的gt，且class_id已重映射
#         """
#         # 过滤本头负责类别
#         mask = torch.zeros(len(targets), dtype=torch.bool, device=targets.device)
#         for cid in self.class_ids:
#             mask |= (targets[:, 1] == cid)
#         targets = targets[mask]

#         if len(targets) == 0:
#             return targets  # 没有本头负责的gt，直接返回空tensor
        
#         # 类别id重映射
#         new_cls = torch.zeros_like(targets[:, 1])
#         for local_id, orig_id in enumerate(self.class_ids):
#             new_cls[targets[:, 1] == orig_id] = local_id
#         targets[:, 1] = new_cls

#         # 坐标缩放
#         targets[:, 2:]  = targets[:, 2:] * scale_tensor  # 从归一化坐标转换到特征图坐标
#         return targets
    
#     def bbox_decode(self, anchor_points, pred_dist):
#         from ultralytics.utils.tal import dist2bbox
#         if self.reg_max == 1:
#             return anchor_points + pred_dist  # 直接回归坐标偏移
#         b, a, c = pred_dist.shape
#         pred_dist = pred_dist.view(
#             b, a, 4, c // 4
#             ).softmax(3).matmul(self.proj.type(pred_dist.dtype))  # 分布式回归转换为坐标
#         return dist2bbox(pred_dist, anchor_points, xywh=False)  # 转换为xyxy格式
    
#     def __call__(self, preds, batch):
#         """
#         args:
#             preds: 本头的原始输出列表（训练模式）
#             batch: 原始batch dict, 包含cls/bboxes/batch_idx

#         returns:
#             (loss_sum, loss_items)
#             loss_items: (box_loss, cls_loss, dfl_loss)
#         """
#         device = preds[0].device
#         self.proj = self.proj.to(device)
#         loss = torch.zeros(3, device=device)  # box_loss, cls_loss, dfl_loss

#         # 构建 target tensor
#         gt_cls = batch['cls'].to(device)  # (n, 1)
#         gt_bboxes = batch['bboxes'].to(device)  # (n, 4)  xywh 归一化
#         gt_idx = batch['batch_idx'].to(device)  # (n,)

#         targets = torch.cat([gt_idx.unsqueeze(1).float(), gt_cls.float(), gt_bboxes], dim=1)  # (n, 6)： [batch_idx, class_id, x_center, y_center, width, height]

#         batch_size = preds[0].shape[0]
#         imgsz = torch.tensor(
#             preds[0].shape[2:], device=device, dtype=gt_bboxes.dtype
#         ) * self.model_head.stride[0]
#         scale_tensor = torch.cat([imgsz[[1, 0]], imgsz[[1, 0]]])

#         # 过滤出本头负责的gt，并重映射类别id
#         # 过滤出本头负责的gt，并重映射类别id
#         targets = self.preprocess(targets, batch_size, scale_tensor)

#         # 生成 anchors
#         anchor_points, stride_tensor = make_anchors(preds, self.model_head.stride, 0.5)

#         # 整理预测
#         pred_distri_list, pred_scores_list = [], []
#         for i, xi in enumerate(preds):
#             bs, _, h, w = xi.shape
#             xi = xi.view(bs, self.no, -1).permute(0, 2, 1)
#             pred_distri_list.append(xi[:, :, :self.reg_max * 4])
#             pred_scores_list.append(xi[:, :, self.reg_max * 4:])

#         pred_distri = torch.cat(pred_distri_list, 1)
#         pred_scores = torch.cat(pred_scores_list, 1)

#         # 解码 bbox
#         pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

#         # 构建 gt_labels, gt_bboxes_xyxy, mask_gt
#         counts = [0 for _ in range(batch_size)]
#         max_gt = 0

#         if len(targets) > 0:
#             for b in range(batch_size):
#                 n = (targets[:, 0] == b).sum().item()
#                 counts[b] = n
#                 max_gt = max(max_gt, n)

#         gt_labels = torch.zeros(batch_size, max_gt, 1, device=device, dtype=torch.long)
#         gt_bboxes_xyxy = torch.zeros(batch_size, max_gt, 4, device=device)
#         mask_gt = torch.zeros(batch_size, max_gt, 1, device=device)

#         if max_gt > 0:
#             for b in range(batch_size):
#                 bmask = targets[:, 0] == b
#                 btgt = targets[bmask]
#                 n = len(btgt)
#                 if n == 0:
#                     continue

#                 gt_labels[b, :n, 0] = btgt[:, 1].long()
#                 gt_bboxes_xyxy[b, :n] = xywh2xyxy(btgt[:, 2:6])
#                 mask_gt[b, :n, 0] = 1.0

#         # 如果本 head 在当前 batch 没有负责的目标：
#         # 仍然计算分类 loss，target 全为背景；box/dfl 不计算。
#         if max_gt == 0:
#             target_scores = torch.zeros_like(pred_scores)
#             loss[1] = self.bce(pred_scores, target_scores).sum() * self.class_weights
#             return loss.sum(), loss.detach()

#         # TAL分配
#         _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
#             pred_scores.detach().sigmoid(),
#             (pred_bboxes.detach() * stride_tensor).type(gt_bboxes_xyxy.dtype),
#             anchor_points * stride_tensor,
#             gt_labels,
#             gt_bboxes_xyxy,
#             mask_gt,
#         )

#         target_scores_sum = target_scores.sum().clamp(min=1)

#         # 分类 loss
#         loss[1] = self.bce(
#             pred_scores, target_scores.to(pred_scores.dtype)
#         ).sum() / target_scores_sum * self.class_weights

#         # 边界框 loss
#         if fg_mask.sum() > 0:
#             target_bboxes /= stride_tensor
#             loss[0], loss[2] = self.bbox_loss(
#                 pred_distri,
#                 pred_bboxes,
#                 anchor_points,
#                 target_bboxes,
#                 target_scores,
#                 target_scores_sum,
#                 fg_mask,
#             )

#         loss[0] *= 7.5
#         loss[2] *= 1.5

#         return loss.sum(), loss.detach()


# class _BBoxLoss(nn.Module):
#     def __init__(self, reg_max):
#         super().__init__()
#         self.reg_max = reg_max
        
#         if reg_max > 1:
#             # self.dfl_loss = v8DetectionLoss.DFLoss(reg_max)
#             self.proj = torch.arange(reg_max + 1, dtype=torch.float)
    
#     def forward(self, pred_dist, pred_bboxes, anchor_points,
#                 target_bboxes, target_scores, target_scores_sum, fg_mask):
#         weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

#         iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

#         if type(iou) is tuple:
#             if len(iou) == 2:
#                 loss_iou = ((1.0 - iou[0]) * iou[1].detach() * weight).sum() / target_scores_sum
#             else:
#                 loss_iou = (iou[0] * iou[1] * weight).sum() / target_scores_sum
#         else:
#             loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

#         # dfl loss
#         if self.reg_max > 1:
#             target_ltrb = self._bbox2dist(
#                 anchor_points, target_bboxes, self.reg_max
#             )

#             loss_dfl = self._df_loss(
#                 pred_dist[fg_mask].view(-1, self.reg_max + 1),
#                 target_ltrb[fg_mask]
#             ) * weight

#             loss_dfl = loss_dfl.sum() / target_scores_sum
#         else:
#             loss_dfl = torch.tensor(0.0, device=pred_dist.device)
        
#         return loss_iou, loss_dfl
    
#     def _bbox2dist(self, anchor_points, bbox, reg_max):
#         x1y1, x2y2 = bbox.chunk(2, -1)
#         return torch.cat(
#             (anchor_points - x1y1, x2y2 - anchor_points), -1
#         ).clamp_(0, reg_max - 0.01)
    
#     def _df_loss(self, pred_dist, target):
#         tl = target.long()
#         tr = tl + 1
#         wl = (tr.float() - target).unsqueeze(-1)
#         wr = (1 - wl)

#         return (
#             nn.functional.cross_entropy(pred_dist, 
#                                         tl.clamp(0, self.reg_max).view(-1),
#                                         reduction='none'
#                                         ).view(tl.shape) * wl.squeeze(-1) 
#             +
#             nn.functional.cross_entropy(pred_dist, 
#                                         tr.clamp(0, self.reg_max).view(-1),
#                                         reduction='none'
#                                         ).view(tr.shape) * wr.squeeze(-1)
#         )


# class DecoupledDetectionLoss():
#     """
#     解耦检测头的联合loss

#     将 LesionLoss 和 FishLoss 分别计算后加权求和

#     args：
#         model : DecoupledDetectModel 实例
#         lesion_cls_weight: lesion头分类loss权重。默认1.5
#         fish_cls_weight: fish头分类loss权重。默认0.5
#         lesion_loss_scale: lesion 总 loss 相对于 fish的缩放系数
#     """
#     def __init__(
#             self, model,
#             lesion_cls_weight: float = 1.5,
#             fish_cls_weight: float = 0.5,
#             lesion_loss_scale: float = 1.0
#     ):
#         from ultralytics.nn.modules.decoupled_head import DetectLesion, DetectFish

#         lesion_head = None
#         fish_head = None
#         for m in model.model.modules():
#             if isinstance(m, DetectLesion):
#                 lesion_head = m
#             if isinstance(m, DetectFish):
#                 fish_head = m
        
#         assert lesion_head is not None, "DetectLesion Module not found"
#         assert fish_head is not None, "DetectFish Moudule not found"

#         # lesion 头：类别 id=[1, 2, 3]
#         self.lesion_loss = _SingleHeadLoss(
#             lesion_head,
#             class_ids=[1, 2, 3],
#             class_weights=lesion_cls_weight
#         )

#         # fish 头：类别 id=[0]内部映射不变
#         self.fish_loss = _SingleHeadLoss(
#             fish_head,
#             class_ids=[0],
#             class_weights=fish_cls_weight
#         )

#         self.lesion_loss_scale = lesion_loss_scale

#     def __call__(self, preds, batch):
#         """
#         args:
#             preds:(lesion_preds, fish_preds) 元组
#                 由 DecoupledDetectionModel.forward 返回
#             batch: 原始batch dict
#         returns:
#             (total_loss, loss_item)
#         """
#         if isinstance(preds, (tuple, list)) and len(preds) == 2 and torch.is_tensor(preds[0]):
#             preds = preds[1]

#         lesion_preds, fish_preds = preds

#         lesion_total, lesion_items = self.lesion_loss(lesion_preds, batch)
#         fish_total, fish_items = self.fish_loss(fish_preds, batch)

#         total = lesion_total * self.lesion_loss_scale + fish_total
#         combined = lesion_items * self.lesion_loss_scale + fish_items
#         return total, combined




"""Robust loss for a two-head fish-body/lesion YOLO detector.

The implementation keeps the training semantics of the existing project:

* lesion head: global classes 1..4, classification gain 1.5;
* fish head: global class 0, classification gain 0.5;
* TAL parameters: top-k 13, alpha 1.0, beta 6.0;
* box and DFL gains: 7.5 and 1.5;
* return value: ``(total_loss, [box_loss, cls_loss, dfl_loss])``.

Unlike a hard-coded implementation, class mappings are read from the head
modules and checked against the model's global ``nc``.  Missing, duplicated,
overlapping, or out-of-range classes therefore fail explicitly instead of
being silently ignored.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.nn.modules.decoupled_head import DetectFish, DetectLesion
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

__all__ = ("DecoupledDetectionLoss",)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when a parallel wrapper is present."""
    return model.module if hasattr(model, "module") else model


def _model_global_nc(model: nn.Module) -> Optional[int]:
    """Read the global class count from common Ultralytics model fields."""
    for candidate in (model, getattr(model, "model", None)):
        if candidate is not None and hasattr(candidate, "nc"):
            value = getattr(candidate, "nc")
            if value is not None:
                return int(value)

    yaml_config = getattr(model, "yaml", None)
    if isinstance(yaml_config, Mapping) and "nc" in yaml_config:
        return int(yaml_config["nc"])
    return None


def _find_single_head(model: nn.Module, head_type: type, name: str) -> nn.Module:
    """Find exactly one head of the requested type."""
    root = getattr(model, "model", model)
    matches = [module for module in root.modules() if isinstance(module, head_type)]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one {} head, but found {}. Check model parsing and "
            "module registration.".format(name, len(matches))
        )
    return matches[0]


def _as_integer_labels(labels: Tensor, context: str) -> Tensor:
    """Validate class labels and return a flattened int64 tensor."""
    labels = labels.reshape(-1)
    if labels.numel() == 0:
        return labels.to(dtype=torch.long)
    if not torch.isfinite(labels).all():
        raise ValueError("{} contains NaN or infinite class labels.".format(context))

    rounded = labels.round()
    if not torch.allclose(labels.float(), rounded.float(), atol=1e-6, rtol=0.0):
        bad = labels[(labels - rounded).abs() > 1e-6][:10].detach().cpu().tolist()
        raise ValueError(
            "{} contains non-integer class labels, for example {}.".format(context, bad)
        )
    return rounded.to(dtype=torch.long)


class _BBoxLoss(nn.Module):
    """IoU and distribution focal loss for one detection head."""

    def __init__(self, distribution_bins: int) -> None:
        super().__init__()
        distribution_bins = int(distribution_bins)
        if distribution_bins <= 1:
            raise ValueError("distribution_bins must be greater than 1.")
        self.distribution_bins = distribution_bins

    @staticmethod
    def _df_loss(pred_dist: Tensor, target: Tensor) -> Tensor:
        """Distribution focal loss for continuous left/right bin targets."""
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.to(target.dtype) - target
        weight_right = 1.0 - weight_left

        loss_left = F.cross_entropy(
            pred_dist, target_left.reshape(-1), reduction="none"
        ).reshape(target_left.shape)
        loss_right = F.cross_entropy(
            pred_dist, target_right.reshape(-1), reduction="none"
        ).reshape(target_right.shape)
        return (
            loss_left * weight_left + loss_right * weight_right
        ).mean(-1, keepdim=True)

    def forward(
        self,
        pred_dist: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        target_bboxes: Tensor,
        target_scores: Tensor,
        target_scores_sum: Tensor,
        foreground_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute foreground-normalized CIoU and DFL losses."""
        if not foreground_mask.any():
            # Connect both zero losses to regression predictions.  This keeps
            # all regression parameters in the autograd graph when a batch has
            # no target for this head, which is important for DDP training.
            zero = pred_dist.sum() * 0.0
            return zero, zero

        weights = target_scores.sum(-1)[foreground_mask].unsqueeze(-1)
        iou_result = bbox_iou(
            pred_bboxes[foreground_mask],
            target_bboxes[foreground_mask],
            xywh=False,
            CIoU=True,
        )
        # Standard Ultralytics returns a tensor.  Some project forks return a
        # tuple containing the IoU tensor and auxiliary quantities.
        iou = iou_result[0] if isinstance(iou_result, (tuple, list)) else iou_result
        iou = iou.squeeze(-1)
        loss_iou = ((1.0 - iou) * weights.squeeze(-1)).sum() / target_scores_sum

        max_distance = self.distribution_bins - 1
        target_ltrb = self._bbox2dist(anchor_points, target_bboxes, max_distance)
        foreground_dist = pred_dist[foreground_mask].reshape(
            -1, self.distribution_bins
        )
        foreground_target = target_ltrb[foreground_mask].clamp_(
            0, max_distance - 0.01
        )
        loss_dfl = self._df_loss(foreground_dist, foreground_target)
        loss_dfl = (loss_dfl * weights).sum() / target_scores_sum
        return loss_iou, loss_dfl

    @staticmethod
    def _bbox2dist(anchor_points: Tensor, bbox: Tensor, max_distance: int) -> Tensor:
        """Transform xyxy boxes into ltrb distances from anchor points."""
        x1y1, x2y2 = bbox.chunk(2, -1)
        return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(
            0, max_distance - 0.01
        )


class _SingleHeadLoss:
    """Loss computation for one task-specific detection head."""

    def __init__(
        self,
        model_head: nn.Module,
        global_class_ids: Sequence[int],
        classification_gain: float,
        box_gain: float = 7.5,
        dfl_gain: float = 1.5,
        tal_topk: int = 13,
        tal_alpha: float = 1.0,
        tal_beta: float = 6.0,
    ) -> None:
        self.head = model_head
        self.nc = int(model_head.nc)
        self.nl = int(model_head.nl)
        self.reg_max = int(model_head.reg_max)
        self.no = int(model_head.no)
        self.stride = model_head.stride

        self.global_class_ids = tuple(int(value) for value in global_class_ids)
        if len(self.global_class_ids) != self.nc:
            raise ValueError(
                "Head nc={} does not match mapping {}.".format(
                    self.nc, self.global_class_ids
                )
            )

        self.classification_gain = float(classification_gain)
        self.box_gain = float(box_gain)
        self.dfl_gain = float(dfl_gain)

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(
            topk=int(tal_topk),
            num_classes=self.nc,
            alpha=float(tal_alpha),
            beta=float(tal_beta),
        )
        self.bbox_loss = _BBoxLoss(self.reg_max)
        self.proj = torch.arange(self.reg_max, dtype=torch.float)

    def _validate_features(self, features: Sequence[Tensor]) -> List[Tensor]:
        """Validate raw Detect feature maps returned during training."""
        if not isinstance(features, (tuple, list)):
            raise TypeError("Head predictions must be a list/tuple of feature tensors.")
        if len(features) != self.nl:
            raise ValueError(
                "Head returned {} feature maps; expected {}.".format(
                    len(features), self.nl
                )
            )
        if not features:
            raise ValueError("Head returned no feature maps.")

        validated = []
        batch_size = None
        for index, feature in enumerate(features):
            if not torch.is_tensor(feature) or feature.ndim != 4:
                raise ValueError(
                    "Feature {} must have shape [B, C, H, W].".format(index)
                )
            if feature.shape[1] != self.no:
                raise ValueError(
                    "Feature {} has {} channels; expected {} (= 4*reg_max + nc).".format(
                        index, feature.shape[1], self.no
                    )
                )
            if feature.shape[2] <= 0 or feature.shape[3] <= 0:
                raise ValueError("Feature {} has invalid spatial dimensions.".format(index))
            if batch_size is None:
                batch_size = feature.shape[0]
            elif feature.shape[0] != batch_size:
                raise ValueError("Detection feature maps have different batch sizes.")
            validated.append(feature)

        stride = self.stride.detach() if torch.is_tensor(self.stride) else torch.as_tensor(self.stride)
        if stride.numel() != self.nl or (stride <= 0).any():
            raise RuntimeError(
                "Detection-head strides are uninitialized or invalid: {}. Build the "
                "Ultralytics model before constructing its criterion.".format(
                    stride.detach().cpu().tolist()
                )
            )
        return validated

    def _select_and_remap_targets(
        self,
        batch: Dict[str, Tensor],
        device: torch.device,
    ) -> Tensor:
        """Select this head's global classes and map them to local IDs.

        Target geometry is deliberately kept in FP32.  During AMP training,
        detection features are commonly FP16, but Ultralytics' TAL assignment
        performs its overlap bookkeeping in FP32.  Converting GT boxes to the
        feature dtype can therefore produce a Float/Half assignment error.
        """
        required = ("batch_idx", "cls", "bboxes")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError("Training batch is missing keys: {}.".format(missing))

        batch_indices = _as_integer_labels(
            batch["batch_idx"].to(device=device), "training batch batch_idx"
        )
        global_labels = _as_integer_labels(
            batch["cls"].to(device=device), "training batch"
        )
        bboxes = batch["bboxes"].reshape(-1, 4).to(
            device=device, dtype=torch.float32
        )

        count = global_labels.numel()
        if batch_indices.numel() != count or bboxes.shape[0] != count:
            raise ValueError(
                "batch_idx, cls, and bboxes contain different target counts: {}, {}, {}.".format(
                    batch_indices.numel(), count, bboxes.shape[0]
                )
            )
        if count == 0:
            return torch.zeros((0, 6), device=device, dtype=torch.float32)
        if not torch.isfinite(bboxes).all():
            raise ValueError("Training batch contains NaN or infinite bounding boxes.")

        max_global_id = max(self.global_class_ids)
        lookup = torch.full(
            (max_global_id + 1,), -1, device=device, dtype=torch.long
        )
        mapping_ids = torch.tensor(
            self.global_class_ids, device=device, dtype=torch.long
        )
        lookup[mapping_ids] = torch.arange(self.nc, device=device, dtype=torch.long)

        in_lookup = (global_labels >= 0) & (global_labels < lookup.numel())
        local_labels = torch.full_like(global_labels, -1)
        local_labels[in_lookup] = lookup[global_labels[in_lookup]]
        selected = local_labels >= 0

        if not selected.any():
            return torch.zeros((0, 6), device=device, dtype=torch.float32)

        return torch.cat(
            (
                batch_indices[selected, None].to(dtype=torch.float32),
                local_labels[selected, None].to(dtype=torch.float32),
                bboxes[selected],
            ),
            dim=1,
        )

    @staticmethod
    def _pack_targets(
        targets: Tensor,
        batch_size: int,
        scale_tensor: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Pack flat YOLO targets into padded per-image tensors."""
        if targets.numel() == 0:
            empty_labels = targets.new_zeros((batch_size, 0, 1))
            empty_boxes = targets.new_zeros((batch_size, 0, 4))
            empty_mask = torch.zeros(
                (batch_size, 0, 1), device=targets.device, dtype=torch.bool
            )
            return empty_labels, empty_boxes, empty_mask

        image_indices = targets[:, 0].long()
        if (image_indices < 0).any() or (image_indices >= batch_size).any():
            bad = image_indices[
                (image_indices < 0) | (image_indices >= batch_size)
            ][:10].detach().cpu().tolist()
            raise ValueError(
                "Targets contain batch indices outside [0, {}): {}.".format(
                    batch_size, bad
                )
            )

        counts = torch.bincount(image_indices, minlength=batch_size)
        max_targets = int(counts.max().item())
        packed = targets.new_zeros((batch_size, max_targets, 5))

        for batch_index in range(batch_size):
            per_image = targets[image_indices == batch_index, 1:]
            if per_image.numel():
                packed[batch_index, : per_image.shape[0]] = per_image

        gt_labels = packed[..., :1]
        gt_bboxes = xywh2xyxy(packed[..., 1:5] * scale_tensor)
        valid_mask = gt_bboxes.sum(-1, keepdim=True).gt_(0)
        return gt_labels, gt_bboxes, valid_mask

    def __call__(
        self,
        predictions: Sequence[Tensor],
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Compute one head's scalar loss and detached three-component vector."""
        features = self._validate_features(predictions)
        device = features[0].device
        dtype = features[0].dtype
        batch_size = features[0].shape[0]

        concatenated = torch.cat(
            [feature.reshape(batch_size, self.no, -1) for feature in features],
            dim=2,
        )
        pred_dist, pred_scores = concatenated.split(
            (self.reg_max * 4, self.nc), dim=1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(features, self.stride, 0.5)
        pred_bboxes = dist2bbox(
            pred_dist.view(batch_size, -1, 4, self.reg_max)
            .softmax(3)
            .matmul(self.proj.to(device=device, dtype=dtype)),
            anchor_points,
            xywh=False,
        )

        image_size = (
            torch.tensor(
                features[0].shape[2:], device=device, dtype=torch.float32
            )
            * self.stride[0].to(device=device, dtype=torch.float32)
        )
        scale_tensor = image_size[[1, 0, 1, 0]]
        targets = self._select_and_remap_targets(batch, device)
        gt_labels, gt_bboxes, mask_gt = self._pack_targets(
            targets, batch_size, scale_tensor
        )

        # Accumulate losses in FP32 even when model activations use FP16.
        loss = torch.zeros(3, device=device, dtype=torch.float32)
        regression_zero = pred_dist.float().sum() * 0.0

        if gt_bboxes.shape[1] == 0:
            target_scores = torch.zeros(
                pred_scores.shape, device=device, dtype=torch.float32
            )
            foreground_mask = torch.zeros(
                pred_scores.shape[:2], device=device, dtype=torch.bool
            )
            target_bboxes = torch.zeros(
                pred_bboxes.shape, device=device, dtype=torch.float32
            )
        else:
            # TAL assignment is a non-gradient operation.  Keep all assignment
            # inputs and computations in FP32 to avoid AMP dtype conflicts in
            # overlap calculation while retaining AMP for the model forward.
            with torch.cuda.amp.autocast(enabled=False):
                _, target_bboxes, target_scores, foreground_mask, _ = self.assigner(
                    pred_scores.detach().float().sigmoid(),
                    (pred_bboxes.detach() * stride_tensor).float(),
                    (anchor_points * stride_tensor).float(),
                    gt_labels.float(),
                    gt_bboxes.float(),
                    mask_gt,
                )

        target_scores_sum = target_scores.sum().clamp(min=1.0)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if foreground_mask.any():
            target_bboxes_grid = target_bboxes / stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_dist,
                pred_bboxes,
                anchor_points,
                target_bboxes_grid,
                target_scores,
                target_scores_sum,
                foreground_mask,
            )
        else:
            loss[0] = regression_zero
            loss[2] = regression_zero

        loss[0] *= self.box_gain
        loss[1] *= self.classification_gain
        loss[2] *= self.dfl_gain
        return loss.sum(), loss.detach()


class DecoupledDetectionLoss:
    """Joint criterion for lesion and fish detection heads."""

    def __init__(
        self,
        model: nn.Module,
        lesion_classification_gain: float = 1.5,
        fish_classification_gain: float = 0.5,
        lesion_loss_scale: float = 1.0,
        fish_loss_scale: float = 1.0,
    ) -> None:
        model = _unwrap_model(model)
        self.lesion_head = _find_single_head(model, DetectLesion, "DetectLesion")
        self.fish_head = _find_single_head(model, DetectFish, "DetectFish")

        self.lesion_class_ids = tuple(self.lesion_head.global_class_ids)
        self.fish_class_ids = tuple(self.fish_head.global_class_ids)
        self.lesion_loss_scale = float(lesion_loss_scale)
        self.fish_loss_scale = float(fish_loss_scale)

        self.global_nc = _model_global_nc(model)
        if self.global_nc is None:
            self.global_nc = max(self.lesion_class_ids + self.fish_class_ids) + 1

        self._validate_class_partition()

        self.lesion_loss = _SingleHeadLoss(
            self.lesion_head,
            self.lesion_class_ids,
            classification_gain=lesion_classification_gain,
        )
        self.fish_loss = _SingleHeadLoss(
            self.fish_head,
            self.fish_class_ids,
            classification_gain=fish_classification_gain,
        )

    def _validate_class_partition(self) -> None:
        """Ensure both heads form a complete, disjoint global class partition."""
        lesion_set = set(self.lesion_class_ids)
        fish_set = set(self.fish_class_ids)
        overlap = sorted(lesion_set.intersection(fish_set))
        if overlap:
            raise ValueError(
                "Fish and lesion heads contain overlapping global classes: {}.".format(
                    overlap
                )
            )

        actual = lesion_set.union(fish_set)
        expected = set(range(int(self.global_nc)))
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                "Decoupled class mapping does not match global nc={}. Missing: {}; "
                "extra: {}. Fish mapping: {}; lesion mapping: {}.".format(
                    self.global_nc,
                    missing,
                    extra,
                    self.fish_class_ids,
                    self.lesion_class_ids,
                )
            )

    def _validate_batch_classes(self, batch: Dict[str, Tensor]) -> None:
        """Reject invalid or unmapped global class labels before filtering."""
        if "cls" not in batch:
            raise KeyError("Training batch is missing the 'cls' field.")
        labels = _as_integer_labels(batch["cls"], "training batch")
        if labels.numel() == 0:
            return
        invalid = (labels < 0) | (labels >= self.global_nc)
        if invalid.any():
            bad = sorted(set(labels[invalid].detach().cpu().tolist()))
            raise ValueError(
                "Training batch contains class IDs outside [0, {}): {}.".format(
                    self.global_nc, bad
                )
            )

    @staticmethod
    def _raw_head_predictions(predictions: object) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
        """Extract ``(lesion_raw, fish_raw)`` from train/validation outputs."""
        value = predictions
        # Ultralytics inference commonly returns (decoded, raw_predictions).
        if (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and torch.is_tensor(value[0])
        ):
            value = value[1]

        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError(
                "Decoupled model must return exactly two raw prediction groups: "
                "(lesion_predictions, fish_predictions)."
            )

        lesion_predictions, fish_predictions = value
        if not isinstance(lesion_predictions, (tuple, list)):
            raise TypeError("Lesion predictions must be a list/tuple of feature maps.")
        if not isinstance(fish_predictions, (tuple, list)):
            raise TypeError("Fish predictions must be a list/tuple of feature maps.")
        return lesion_predictions, fish_predictions

    def __call__(
        self,
        predictions: object,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Compute the weighted joint loss and detached component totals."""
        self._validate_batch_classes(batch)
        lesion_predictions, fish_predictions = self._raw_head_predictions(predictions)

        lesion_total, lesion_items = self.lesion_loss(
            lesion_predictions, batch
        )
        fish_total, fish_items = self.fish_loss(fish_predictions, batch)

        total = (
            self.lesion_loss_scale * lesion_total
            + self.fish_loss_scale * fish_total
        )
        combined_items = (
            self.lesion_loss_scale * lesion_items
            + self.fish_loss_scale * fish_items
        )
        return total, combined_items



