"""
`sumix` data augmentation proposed in the BirdCLEF 2023 competition.
https://www.kaggle.com/c/birdclef-2023/discussion/412922
"""
import torch

def sumix(waves: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, max_percent: float = 1.0, min_percent: float = 0.3):
    batch_size = len(labels)
    perm = torch.randperm(batch_size)
    coeffs_1 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (
        max_percent  - min_percent
    ) + min_percent
    coeffs_2 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (
        max_percent  - min_percent
    ) + min_percent
    label_coeffs_1 = torch.where(coeffs_1 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_1))
    label_coeffs_2 = torch.where(coeffs_2 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_2))
    labels = label_coeffs_1 * labels + label_coeffs_2 * labels[perm]
    labels = torch.clip(labels, 0, 1)

    waves = coeffs_1 * waves + coeffs_2 * waves[perm]
    weights = coeffs_1.ravel() * weights + coeffs_2.ravel()  * weights[perm]
    return waves, labels, weights


def sumup(waves: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    batch_size = len(labels)
    perm = torch.randperm(batch_size)

    waves = waves + waves[perm]
    labels = torch.clip(labels + labels[perm], min=0, max=1)
    weights = weights + weights[perm]

    return waves, labels, weights
