"""
MSQ-style LSB regularization for bit-level weight sparsity.

Uses a sinusoidal penalty that has minima at coarse quantization grid points
(multiples of 2^(-qf+k)), encouraging the k least significant bits of
quantized weights to become zero.

Reference: MSQ (ICCV 2025) - Memory-Efficient Bit Sparsification Quantization
"""

import math
import torch


def compute_lsb_penalty(model, qf, k=2, target_layers='rnn'):
    """
    Compute differentiable LSB regularization penalty on weight parameters.

    Uses sin^2(pi * w / coarse_step) which has minima at coarse grid points
    (multiples of 2^(-qf+k)). This smoothly pushes weights toward values
    whose lower k bits are zero in the Q format.

    Args:
        model: PyTorch model
        qf: Number of fractional bits (e.g., 7 for Q1.7)
        k: Number of LSBs to penalize (default 2)
        target_layers: Which layers to regularize ('rnn', 'all', or 'fc')

    Returns:
        Scalar penalty tensor (differentiable)
    """
    coarse_step = 1.0 / (2 ** (qf - k))  # Grid spacing when lower k bits are zero
    scale = math.pi / coarse_step

    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    n_params = 0

    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        if target_layers == 'rnn' and 'rnn' not in name:
            continue
        if target_layers == 'fc' and 'fc' not in name:
            continue

        # sin^2 penalty: minima at multiples of coarse_step
        penalty = penalty + torch.mean(torch.sin(scale * param) ** 2)
        n_params += 1

    if n_params > 0:
        penalty = penalty / n_params

    return penalty
