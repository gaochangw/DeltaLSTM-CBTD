"""
Recurrent Vision Transformer (RVT) for event-based eye tracking.

Ported from RD211/ERVT (https://github.com/RD211/ERVT).
Architecture: Conv -> LayerNorm -> FullSelfAttention -> MLP -> DWSConvLSTM2d per stage,
with a linear head for (x, y) regression.

Reference:
  Gehrig & Scaramuzza, "Recurrent Vision Transformers for Object Detection
  with Event Cameras", CVPR 2023. arXiv:2212.05598
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


class SelfAttentionCl(nn.Module):
    """Channels-last multi-head self-attention (B, ..., C)."""

    def __init__(self, dim: int, dim_head: int = 32, bias: bool = True):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        q, k, v = (self.qkv(x)
                    .view(B, -1, self.num_heads, self.dim_head * 3)
                    .transpose(1, 2)
                    .chunk(3, dim=3))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        return x


class FullSelfAttention(nn.Module):
    """Full self-attention augmented with LayerScale."""

    def __init__(self, channels: int, dim_head: int = 32):
        super().__init__()
        self.channels = channels
        self.ls = LayerScale(channels)
        self.mhsa = SelfAttentionCl(dim=channels, dim_head=dim_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.size()
        attn_output = self.mhsa(x.view(B, H * W, C)).view(B, H, W, C)
        attn_output = self.ls(attn_output)
        return attn_output


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
    """Depthwise-separable convolutional LSTM in NCHW format."""

    def __init__(self, dim: int, dws_conv_kernel_size: int = 7,
                 cell_update_dropout: float = 0.2):
        super().__init__()
        self.dim = dim

        xh_dim = dim * 2
        gates_dim = dim * 4
        self.conv3x3_dws = nn.Conv2d(
            in_channels=xh_dim, out_channels=xh_dim,
            kernel_size=dws_conv_kernel_size,
            padding=dws_conv_kernel_size // 2,
            groups=xh_dim)
        self.conv1x1 = nn.Conv2d(
            in_channels=xh_dim, out_channels=gates_dim, kernel_size=1)
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

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

        gates, cell_input = torch.tensor_split(mix, [self.dim * 3], dim=1)
        gates = torch.sigmoid(gates)
        forget_gate, input_gate, output_gate = torch.tensor_split(gates, 3, dim=1)

        cell_input = self.cell_update_dropout(torch.tanh(cell_input))

        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * torch.tanh(c_t)

        return h_t, c_t


class RVTBlock(nn.Module):
    """One RVT stage: Conv -> LN -> FSA -> LN -> MLP -> ConvLSTM."""

    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int, stride: int, dim_head: int = 32,
                 expansion_ratio: int = 2, drop_prob: float = 0.6):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2)
        self.ln1 = nn.LayerNorm(output_channels)
        self.fsa = FullSelfAttention(channels=output_channels, dim_head=dim_head)
        self.ln2 = nn.LayerNorm(output_channels)
        self.mlpb = MLP(dim=output_channels, channel_last=True,
                        expansion_ratio=expansion_ratio, gated=True,
                        bias=True, drop_prob=drop_prob)
        self.lstm = DWSConvLSTM2d(dim=output_channels)

    def forward(self, x: torch.Tensor,
                h: Optional[torch.Tensor],
                c: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv(x)
        x_conv = x_conv.permute(0, 2, 3, 1)  # NCHW -> NHWC for LayerNorm
        x_conv = self.ln1(x_conv)

        x_bsa = self.fsa(x_conv)
        x_bsa = x_bsa + x_conv  # residual
        x_bsa = self.ln2(x_bsa)
        x_bsa = self.mlpb(x_bsa)

        x_out = x_bsa.permute(0, 3, 1, 2)  # NHWC -> NCHW for ConvLSTM
        x_out, c = self.lstm(x_out, (h, c))

        return x_out, c


class RVT(nn.Module):
    """Recurrent Vision Transformer.

    Args:
        in_channels: Input channels (e.g. 3 for n_time_bins).
        stages: List of dicts, each with keys:
            output_channels, kernel_size, stride
        dim_head: Attention head dimension.
        drop_prob: MLP dropout probability.
        expansion_ratio: MLP expansion ratio.
        input_height: Input spatial height (e.g. 60).
        input_width: Input spatial width (e.g. 80).
    """

    def __init__(self, in_channels: int, stages: list,
                 dim_head: int = 32, drop_prob: float = 0.6,
                 expansion_ratio: int = 2,
                 input_height: int = 60, input_width: int = 80):
        super().__init__()

        self.stage_modules = nn.ModuleList()
        prev_channels = in_channels
        for cfg in stages:
            block = RVTBlock(
                input_channels=prev_channels,
                output_channels=cfg['output_channels'],
                kernel_size=cfg['kernel_size'],
                stride=cfg['stride'],
                dim_head=dim_head,
                expansion_ratio=expansion_ratio,
                drop_prob=drop_prob,
            )
            self.stage_modules.append(block)
            prev_channels = cfg['output_channels']

        # Compute flattened dimension for the linear head
        stride_factor = reduce(lambda a, b: a * b, [s['stride'] for s in stages])
        head_dim = int(stages[-1]['output_channels']
                       * (input_height // stride_factor)
                       * (input_width // stride_factor))
        self.linear_head = nn.Linear(head_dim, 2)

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

            xt_flat = torch.flatten(xt, 1)
            out = self.linear_head(xt_flat)
            outputs.append(out)

        coordinates = torch.stack(outputs, dim=1)  # (B, T, 2)
        return coordinates, lstm_states
