from typing import Optional, Tuple
import torch
import torch.nn as nn
from utils import util
from nnmodels.deltalstm import DeltaRNN


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

    def forward(self, input, state: Optional[Tuple] = None, feature_lengths=None):
        # Attributes
        if self.training:
            quantize = 0
        else:
            quantize = self.qa

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