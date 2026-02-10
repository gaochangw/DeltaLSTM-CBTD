from typing import Optional, Tuple
import torch
import torch.nn as nn
from utils import util
from nnmodels.deltalstm import DeltaRNN
from nnmodels.rvt import RVT
from nnmodels.janeyenet import JaneEyeNet


class Model(nn.Module):
    def __init__(self, args, input_size, n_classes):
        super(Model, self).__init__()
        self.inp_size = input_size
        self.hid_type = args.hid_type
        self.hid_size = args.hid_size
        self.hid_layers = args.hid_layers
        self.hid_dropout = args.hid_dropout
        self.fc_type = args.fc_type
        self.fc_extra_size = args.fc_extra_size
        self.fc_dropout = args.hid_dropout
        self.qa = args.qa
        self.qw = args.qw
        self.aqi = args.aqi
        self.aqf = args.aqf
        self.wqi = args.wqi
        self.wqf = args.wqf
        self.nqi = args.nqi
        self.nqf = args.nqf
        self.qa_fc_extra = args.qa_fc_extra
        self.aqi_fc_extra = args.aqi_fc_extra
        self.aqf_fc_extra = args.aqf_fc_extra
        self.qa_fc_final = args.qa_fc_final
        self.aqi_fc_final = args.aqi_fc_final
        self.aqf_fc_final = args.aqf_fc_final
        self.th_x = util.quantize_array(args.th_x, self.aqi, self.aqf, self.qa)
        self.th_h = util.quantize_array(args.th_h, self.aqi, self.aqf, self.qa)
        self.use_cuda = args.use_cuda
        self.hardsigmoid = args.hardsigmoid
        self.hardtanh = args.hardtanh

        if 'num_array' in args:
            self.num_array = args.num_array
        if 'num_array_pe' in args:
            self.num_array_pe = args.num_array_pe

        # CNN frontend for spatial feature extraction (e.g., 3ET+ voxel grids)
        # RVT and JaneEyeNet have their own conv stages; skip CNN frontend
        self.cnn_frontend = getattr(args, 'cnn_frontend', 0) if self.hid_type not in ('RVT', 'JaneEyeNet') else 0
        if self.cnn_frontend:
            n_channels = input_size  # n_time_bins (e.g., 3)
            pool_h = getattr(args, 'cnn_pool_height', 5)
            pool_w = getattr(args, 'cnn_pool_width', 4)
            self.cnn = nn.Sequential(
                nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((pool_h, pool_w)),
            )
            # Compute CNN output size via dummy forward pass
            cnn_input_height = getattr(args, 'cnn_input_height', 60)
            cnn_input_width = getattr(args, 'cnn_input_width', 80)
            self.cnn_input_height = cnn_input_height
            self.cnn_input_width = cnn_input_width
            with torch.no_grad():
                dummy = torch.zeros(1, n_channels, cnn_input_height, cnn_input_width)
                dummy_out = self.cnn(dummy)
                self.cnn_output_size = dummy_out.numel() // dummy_out.shape[0]
            self.inp_size = self.cnn_output_size

        # Instantiate RNN layers
        if self.hid_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.inp_size,
                               hidden_size=self.hid_size,
                               num_layers=self.hid_layers,
                               bias=True,
                               bidirectional=False,
                               dropout=self.hid_dropout)
        elif self.hid_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.inp_size,
                              hidden_size=self.hid_size,
                              num_layers=self.hid_layers,
                              bias=True,
                              bidirectional=False,
                              dropout=self.hid_dropout)
        elif self.hid_type == 'DeltaLSTM':
            self.rnn = DeltaRNN(mode='DeltaLSTM',
                                input_size=self.inp_size,
                                hidden_size=self.hid_size,
                                num_layers=self.hid_layers,
                                th_x=self.th_x,
                                th_h=self.th_h,
                                aqi=self.aqi,
                                aqf=self.aqf,
                                nqi=self.nqi,
                                nqf=self.nqf,
                                num_array=self.num_array,
                                num_array_pe=self.num_array_pe,
                                hardtanh=self.hardtanh,
                                hardsigmoid=self.hardsigmoid)
        elif self.hid_type == 'RVT':
            # RVT replaces CNN + RNN + FC entirely
            rvt_channels = [int(c) for c in getattr(args, 'rvt_channels', '32,64').split(',')]
            rvt_dim_head = getattr(args, 'rvt_dim_head', 32)
            rvt_drop_prob = getattr(args, 'rvt_drop_prob', 0.6)
            rvt_expansion_ratio = getattr(args, 'rvt_expansion_ratio', 2)
            input_height = getattr(args, 'cnn_input_height', 60)
            input_width = getattr(args, 'cnn_input_width', 80)
            in_channels = getattr(args, 'n_time_bins', input_size)

            stages = []
            kernel_sizes = [7, 3]  # stage 1: 7x7, stage 2+: 3x3
            for i, out_ch in enumerate(rvt_channels):
                stages.append({
                    'output_channels': out_ch,
                    'kernel_size': kernel_sizes[i] if i < len(kernel_sizes) else 3,
                    'stride': 2,
                })

            self.rvt = RVT(
                in_channels=in_channels,
                stages=stages,
                dim_head=rvt_dim_head,
                drop_prob=rvt_drop_prob,
                expansion_ratio=rvt_expansion_ratio,
                input_height=input_height,
                input_width=input_width,
            )
            # Skip FC layers for RVT (CNN frontend already skipped above)
            return

        elif self.hid_type == 'JaneEyeNet':
            # JaneEye-Net: simplified RVT with 3-conv blocks, no attention
            jen_kernel_size = getattr(args, 'jen_kernel_size', 3)
            jen_output_channels = getattr(args, 'jen_output_channels', 32)
            jen_stride = getattr(args, 'jen_stride', 2)
            jen_drop_prob = getattr(args, 'jen_drop_prob', 0.6)
            jen_expansion_ratio = getattr(args, 'jen_expansion_ratio', 2)
            input_height = getattr(args, 'cnn_input_height', 60)
            input_width = getattr(args, 'cnn_input_width', 80)
            in_channels = getattr(args, 'n_time_bins', input_size)

            stages = [{
                'output_channels': jen_output_channels,
                'kernel_size': jen_kernel_size,
                'stride': jen_stride,
            }]

            self.janeyenet = JaneEyeNet(
                in_channels=in_channels,
                stages=stages,
                drop_prob=jen_drop_prob,
                expansion_ratio=jen_expansion_ratio,
                input_height=input_height,
                input_width=input_width,
            )
            # Skip RNN/FC layers
            return

        # Instantiate FC layers
        if self.fc_extra_size != 0:
            self.fc_extra = nn.Sequential(
                nn.Linear(in_features=self.hid_size, out_features=self.fc_extra_size, bias=True),
                nn.ReLU(),
                nn.Dropout(p=self.fc_dropout)
            )
            self.fc_final = nn.Linear(in_features=self.fc_extra_size, out_features=n_classes, bias=True)
        else:
            self.fc_final = nn.Linear(in_features=self.hid_size, out_features=n_classes, bias=True)

    def set_qa_fc_final(self, x):
        """
        Set quantization of activation
        :param x: If x == 1, quantize activation; else, don't quantize.
        :return:
        """
        self.qa_fc_final = x

    def cnn_forward(self, input):
        """CNN frontend for spatial feature extraction.

        Args:
            input: (T, B, C, H, W) tensor

        Returns:
            (T, B, F) tensor where F is the flattened CNN output size
        """
        T, B, C, H, W = input.shape
        # Reshape to (T*B, C, H, W) for batch processing
        x = input.reshape(T * B, C, H, W)
        # CNN forward
        x = self.cnn(x)
        # Flatten spatial dims: (T*B, F)
        x = x.reshape(T * B, -1)
        # Reshape back to (T, B, F)
        x = x.reshape(T, B, -1)
        return x

    def rvt_forward(self, input, state=None, feature_lengths=None):
        """Forward pass for RVT model.

        Args:
            input: (T, B, C, H, W) tensor from dataloader.
            state: LSTM states (list of (h,c) tuples) or None.
            feature_lengths: unused for RVT.

        Returns:
            outputs: (out_final, None) where out_final is (B, T, 2).
            state: updated LSTM states.
            reg: zero tensor (no regularization).
        """
        # Transpose from (T, B, C, H, W) to (B, T, C, H, W)
        x = input.permute(1, 0, 2, 3, 4)
        coordinates, lstm_states = self.rvt(x, state)
        reg = input.new_zeros(())
        return (coordinates, None), lstm_states, reg

    def janeyenet_forward(self, input, state=None, feature_lengths=None):
        """Forward pass for JaneEye-Net model.

        Args:
            input: (T, B, C, H, W) tensor from dataloader.
            state: LSTM states (list of (h,c) tuples) or None.
            feature_lengths: unused for JaneEyeNet.

        Returns:
            outputs: (out_final, None) where out_final is (B, T, 2).
            state: updated LSTM states.
            reg: zero tensor (no regularization).
        """
        # Transpose from (T, B, C, H, W) to (B, T, C, H, W)
        x = input.permute(1, 0, 2, 3, 4)
        coordinates, lstm_states = self.janeyenet(x, state)
        reg = input.new_zeros(())
        return (coordinates, None), lstm_states, reg

    def forward(self, input, state: Optional[Tuple] = None, feature_lengths=None):
        # RVT path: bypasses CNN, RNN, and FC entirely
        if self.hid_type == 'RVT':
            return self.rvt_forward(input, state, feature_lengths)

        # JaneEyeNet path: bypasses CNN, RNN, and FC entirely
        if self.hid_type == 'JaneEyeNet':
            return self.janeyenet_forward(input, state, feature_lengths)

        # Attributes
        if self.training:
            quantize = 0
        else:
            quantize = self.qa

        # CNN Frontend (if enabled)
        if self.cnn_frontend:
            input = self.cnn_forward(input)

        # Flatten RNN Parameters if possible
        if self.hid_type in ['GRU', 'LSTM']:
            self.rnn.flatten_parameters()

        # RNN Forward
        if feature_lengths is not None:
            out_rnn, state, reg = self.rnn_forward(input, state, quantize, feature_lengths)
        else:
            out_rnn, state, reg = self.rnn_forward(input, state, quantize)

        # FC Forward
        out_final = self.fc_forward(out_rnn, quantize)

        outputs = (out_final, out_rnn)

        return outputs, state, reg

    def rnn_forward(self, input, state, quantize, feature_lengths=None):
        if 'Delta' in self.hid_type:
            out, state, reg = self.rnn(input, state, quantize, feature_lengths)
        else:
            out, state = self.rnn(input, state)
            reg = None
        out = out.transpose(0, 1)  # Transpose RNN Output to (N, T, H)
        return out, state, reg

    def fc_forward(self, input, quantize):
        if self.fc_extra_size:
            out_fc = self.fc_extra(input)
            out_fc = util.quantize_tensor(out_fc, self.aqi, self.aqf, quantize)
            out_fc = self.fc_final(out_fc)
        else:
            out_fc = self.fc_final(input)
        out = util.quantize_tensor(out_fc, self.aqi_fc_final, self.aqf_fc_final, quantize)
        return out