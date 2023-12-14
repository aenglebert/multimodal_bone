import torch
from torch import nn


class SigLIPLoss(nn.Module):
    def __init__(self, temperature: float = 10.0, bias: float = -10.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, similarity: torch.Tensor, targets=None) -> torch.Tensor:
        logits = similarity * self.temperature + self.bias
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
