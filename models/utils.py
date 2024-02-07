from typing import Optional, Sequence, Tuple, Union

from einops import rearrange

import numpy as np

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


import bitsandbytes as bnb

optimizer_dict = {
    "AdamW": bnb.optim.AdamW,
    "AdamW8bit": bnb.optim.AdamW8bit,
    "Adam": bnb.optim.Adam,
    "Adam8bit": bnb.optim.Adam8bit,
    "SGD": bnb.optim.SGD,
    "SGD8bit": bnb.optim.SGD8bit,
    "Lion": bnb.optim.Lion,
    "Lion8bit": bnb.optim.Lion8bit,
}


class SigLIPLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, similarity: torch.Tensor, targets=None) -> torch.Tensor:
        return siglip_loss(similarity, targets, self.temperature)


def siglip_loss(similarity: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    logits = similarity / temperature
    n = len(logits)

    if targets is None:
        targets = torch.eye(n, device=logits.device)

    labels = 2 * targets - 1
    return -torch.sum(nn.functional.logsigmoid(labels * logits)) / n


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    logits = similarity / temperature
    caption_loss = contrastive_loss(logits)
    image_loss = contrastive_loss(logits.t())
    return (caption_loss + image_loss) / 2.0


class ClipLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, similarity: torch.Tensor) -> torch.Tensor:
        return clip_loss(similarity, self.temperature)


def pi_resize_patch_embed(
    patch_embed: Tensor,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """
    Function reused from https://github.com/bwconrad/flexivit/blob/main/flexivit_pytorch/patch_embed.py
    Resample patch embedding weights to a target resolution via pseudo-inverse
    resizing.

    Based on:
        https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py
        https://arxiv.org/abs/2212.08013

    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    old_patch_size = tuple(patch_embed.shape[2:])

    # Return original kernel if no resize is necessary
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: Tensor, shape: Tuple[int, int]):
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=interpolation,
            antialias=antialias,
        )
        return x_resized[0, 0, ...]

    def calculate_pinv(old_shape: Tuple[int, int], new_shape: Tuple[int, int]):
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    # Calculate pseudo-inverse of resize matrix
    resize_matrix_pinv = calculate_pinv(old_patch_size, new_patch_size)
    resize_matrix_pinv = resize_matrix_pinv.to(patch_embed.device)

    def resample_patch_embed(patch_embed: Tensor):
        h, w = new_patch_size
        resampled_kernel = resize_matrix_pinv @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

    v_resample_patch_embed = torch.vmap(torch.vmap(resample_patch_embed, 0, 0), 1, 1)

    return v_resample_patch_embed(patch_embed)