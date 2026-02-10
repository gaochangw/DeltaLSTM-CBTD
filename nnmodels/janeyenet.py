"""
JaneEye-Net: Simplified Recurrent Vision Transformer for event-based eye tracking.

Ported from lab-emi/JaneEye-Net (model/RVT.py).

Key differences from the existing RVT (nnmodels/rvt.py):
  - RVTBlock: 3 Conv layers (7x7 -> 3x3 -> 3x3) + ReLU, NO attention, NO LayerNorm
  - DWSConvLSTM2d: Simplified forget-gate only, c = f*c_prev + (1-f)*cell_input, h = c
  - Detection head: GlobalMaxPool2d(1) -> Linear(out_ch, 2)
"""

import math
from functools import reduce
from typing import Optional, Tuple, List

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """Learnable per-channel scaling (init 1e-5)."""

    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class GLU(nn.Module):
    """Gated Linear Unit with GELU activation (GeGLU)."""

    def __init__(self, dim_in: int, dim_out: int, channel_last: bool, bias: bool = True):
        super().__init__()
        proj_out_dim = dim_out * 2
        self.proj = (nn.Linear(dim_in, proj_out_dim, bias=bias) if channel_last
                     else nn.Conv2d(dim_in, proj_out_dim, kernel_size=1, stride=1, bias=bias))
        self.channel_dim = -1 if channel_last else 1
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = torch.tensor_split(self.proj(x), 2, dim=self.channel_dim)
        return x * self.gelu(gate)


class MLP(nn.Module):
    """Gated MLP with configurable expansion ratio and dropout."""

    def __init__(self, dim: int, channel_last: bool, expansion_ratio: int,
                 gated: bool = True, bias: bool = True, drop_prob: float = 0.0):
        super().__init__()
        inner_dim = int(dim * expansion_ratio)
        if gated:
            inner_dim = math.floor(inner_dim * 2 / 3 / 32) * 32  # multiple of 32
            proj_in = GLU(dim_in=dim, dim_out=inner_dim, channel_last=channel_last, bias=bias)
        else:
            proj_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=bias) if channel_last
                else nn.Conv2d(dim, inner_dim, kernel_size=1, stride=1, bias=bias),
                nn.GELU(),
            )
        self.net = nn.Sequential(
            proj_in,
            nn.Dropout(p=drop_prob),
            nn.Linear(inner_dim, dim, bias=bias) if channel_last
            else nn.Conv2d(inner_dim, dim, kernel_size=1, stride=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DWSConvLSTM2d(nn.Module):
    """Simplified depthwise-separable convolutional LSTM (forget-gate only).

    Differences from standard ConvLSTM:
      - Only a forget gate (no input/output gates)
      - Cell update: c = f * c_prev + (1 - f) * cell_input
      - Hidden state: h = c (direct copy)
    """

    def __init__(self, dim: int, dws_conv_kernel_size: int = 7,
                 cell_update_dropout: float = 0.2):
        super().__init__()
        self.dim = dim

        xh_dim = dim * 2
        gates_dim = dim * 1  # Only forget gate

        self.conv3x3_dws = nn.Conv2d(
            in_channels=xh_dim, out_channels=xh_dim,
            kernel_size=dws_conv_kernel_size,
            padding=dws_conv_kernel_size // 2,
            groups=xh_dim)
        self.conv1x1 = nn.Conv2d(
            in_channels=xh_dim,
            out_channels=gates_dim + dim,  # forget gate + cell input
            kernel_size=1)

    def forward(self, x: torch.Tensor,
                h_and_c_previous: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h_and_c_previous[0] is None:
            hidden = torch.zeros_like(x)
            cell = torch.zeros_like(x)
            h_and_c_previous = (hidden, cell)

        h_tm1, c_tm1 = h_and_c_previous

        xh = torch.cat((x, h_tm1), dim=1)
        xh = self.conv3x3_dws(xh)
        mix = self.conv1x1(xh)

        # Split: forget gate | cell input
        forget_gate, cell_input = torch.tensor_split(mix, [self.dim], dim=1)
        forget_gate = torch.sigmoid(forget_gate)
        cell_input = torch.tanh(cell_input)

        # Simplified cell update
        c_t = forget_gate * c_tm1 + (1 - forget_gate) * cell_input
        # Hidden state = cell state
        h_t = c_t

        return h_t, c_t


class JaneEyeNetBlock(nn.Module):
    """One JaneEye-Net stage: Conv1(7x7) -> Conv2(KxK) -> Conv3(KxK) -> MLP -> ConvLSTM.

    No attention, no LayerNorm — purely convolutional with ReLU activations.
    """

    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int, stride: int,
                 expansion_ratio: int = 2, drop_prob: float = 0.6):
        super().__init__()
        # 3 conv layers as in JaneEye-Net
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=7,
                               stride=stride, padding=7 // 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(16, output_channels, kernel_size=kernel_size,
                               stride=1, padding=1)

        self.mlpb = MLP(dim=output_channels, channel_last=True,
                        expansion_ratio=expansion_ratio, gated=True,
                        bias=True, drop_prob=drop_prob)

        self.lstm = DWSConvLSTM2d(dim=output_channels)

    def forward(self, x: torch.Tensor,
                h: Optional[torch.Tensor],
                c: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Conv1 + ReLU
        x = self.conv1(x)
        x = torch.relu(x)

        # Conv2 + ReLU
        x = self.conv2(x)
        x = torch.relu(x)

        # Conv3 + ReLU
        x = self.conv3(x)
        x = torch.relu(x)

        # MLP (channels-last)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.mlpb(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # ConvLSTM
        x, c = self.lstm(x, (h, c))

        return x, c


class JaneEyeNet(nn.Module):
    """JaneEye-Net: simplified RVT for event-based eye tracking.

    Uses GlobalMaxPool2d + Linear(out_ch, 2) detection head instead of
    flatten + Linear(head_dim, 2).

    Args:
        in_channels: Input channels (e.g. 3 for n_time_bins).
        stages: List of dicts with keys: output_channels, kernel_size, stride.
        drop_prob: MLP dropout probability.
        expansion_ratio: MLP expansion ratio.
        input_height: Input spatial height (e.g. 60).
        input_width: Input spatial width (e.g. 80).
    """

    def __init__(self, in_channels: int, stages: list,
                 drop_prob: float = 0.6, expansion_ratio: int = 2,
                 input_height: int = 60, input_width: int = 80):
        super().__init__()

        self.stage_modules = nn.ModuleList()
        prev_channels = in_channels
        for cfg in stages:
            block = JaneEyeNetBlock(
                input_channels=prev_channels,
                output_channels=cfg['output_channels'],
                kernel_size=cfg['kernel_size'],
                stride=cfg['stride'],
                expansion_ratio=expansion_ratio,
                drop_prob=drop_prob,
            )
            self.stage_modules.append(block)
            prev_channels = cfg['output_channels']

        # GlobalMaxPool2d + Linear head (like JaneEye-Net)
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.linear_head = nn.Linear(stages[-1]['output_channels'], 2)

    def forward(self, x: torch.Tensor,
                lstm_states: Optional[List[Tuple[Optional[torch.Tensor],
                                                  Optional[torch.Tensor]]]] = None
                ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (B, T, C, H, W) input tensor.
            lstm_states: List of (h, c) tuples per stage, or None.

        Returns:
            coordinates: (B, T, 2) predictions.
            lstm_states: Updated list of (h, c) tuples.
        """
        B, T = x.shape[0], x.shape[1]

        if lstm_states is None:
            lstm_states = [(None, None)] * len(self.stage_modules)

        outputs = []
        for t in range(T):
            xt = x[:, t, :, :, :]  # (B, C, H, W)

            for i, stage in enumerate(self.stage_modules):
                h_prev, c_prev = lstm_states[i]
                xt, c = stage(xt, h_prev, c_prev)
                lstm_states[i] = (xt, c)

            # Global max pooling
            xt = self.global_pool(xt)  # (B, C, 1, 1)
            xt = torch.flatten(xt, 1)  # (B, C)

            out = self.linear_head(xt)
            outputs.append(out)

        coordinates = torch.stack(outputs, dim=1)  # (B, T, 2)
        return coordinates, lstm_states
