"""Model factory (timm) + param groups cho discriminative LR."""
from __future__ import annotations

import timm
import torch.nn as nn


def build_model(arch: str, pretrained: bool = True, drop_rate: float = 0.0) -> nn.Module:
    """Backbone timm bất kỳ, 1 logit đầu ra (binary — Tầng 1 phát hiện)."""
    return timm.create_model(arch, pretrained=pretrained, num_classes=1, drop_rate=drop_rate)


def build_classifier(arch: str, num_classes: int, in_chans: int = 3,
                     pretrained: bool = True, drop_rate: float = 0.0) -> nn.Module:
    """Backbone timm đa lớp, `in_chans` kênh (Tầng 2 — phân loại đa thì MRI).

    timm tự thích ứng trọng số conv1 khi in_chans != 3 (lặp/chia trung bình)."""
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes,
                             in_chans=in_chans, drop_rate=drop_rate)


def param_groups(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float):
    """Tách head vs backbone để đặt LR khác nhau (discriminative LR)."""
    try:
        head = model.get_classifier()
        head_ids = {id(p) for p in head.parameters()}
    except Exception:  # noqa: BLE001
        head_ids = set()
    head_p, back_p = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (head_p if id(p) in head_ids else back_p).append(p)
    groups = [{"params": back_p, "lr": lr_backbone}]
    if head_p:
        groups.append({"params": head_p, "lr": lr_head})
    return groups, weight_decay


def set_backbone_requires_grad(model: nn.Module, flag: bool) -> None:
    """Freeze/unfreeze backbone (giữ head luôn trainable)."""
    try:
        head_ids = {id(p) for p in model.get_classifier().parameters()}
    except Exception:  # noqa: BLE001
        head_ids = set()
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = flag
