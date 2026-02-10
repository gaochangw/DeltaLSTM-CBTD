import math
from typing import Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import utils.util as util


def hsigmoid(x):
    """
    Computes element-wise hard sigmoid of x.
    See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    """
    x = (0.25 * x) + 0.5
    x = F.threshold(-x, -1.0, -1.0)
    x = F.threshold(-x, 0.0, 0.0)
    return x


class DeltaRNN(nn.Module):
    def __init__(self, mode: str, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False, **kwargs):
        super(DeltaRNN, self).__init__()
        # Attributes
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.aqi = kwargs['aqi']
        self.aqf = kwargs['aqf']
        self.nqi = kwargs['nqi']
        self.nqf = kwargs['nqf']
        self.th_x = kwargs['th_x']
        self.th_h = kwargs['th_h']
        self.num_array = kwargs['num_array']
        self.num_array_pe = kwargs['num_array_pe']
        self.hardsigmoid = kwargs['hardsigmoid']
        self.hardtanh = kwargs['hardtanh']

        # Adapt state sizes to hardware supported size
        self.adapted_input_size = int(math.ceil(input_size / float(self.num_array)) * self.num_array)
        self.adapted_hidden_size = int(math.ceil(hidden_size / float(self.num_array)) * self.num_array)
        self.diff_input_size = self.adapted_input_size - self.input_size
        self.diff_hidden_size = self.adapted_hidden_size - self.hidden_size

        # Initialize Swap Lists
        self.input_swap_list = np.arange(0, self.adapted_input_size, dtype=int)
        self.hidden_swap_list = []
        for i in range(0, num_layers):
            self.hidden_swap_list.append(np.arange(0, self.adapted_hidden_size))

        # Calculate gate size for initializing state memory
        self.num_gates = 4
        self.num_states = 5
        self.gate_size = self.num_gates * self.hidden_size
        self.adapted_gate_size = self.num_gates * self.adapted_hidden_size

        # Instantiate parameters
        for layer in range(num_layers):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size
            w_ih = Parameter(torch.Tensor(self.gate_size, layer_input_size))
            w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
            b_ih = Parameter(torch.Tensor(self.gate_size))
            b_hh = Parameter(torch.Tensor(self.gate_size))
            layer_params = (w_ih, w_hh, b_ih, b_hh)

            param_names = ['weight_ih_l{}', 'weight_hh_l{}']
            if bias:
                param_names += ['bias_ih_l{}', 'bias_hh_l{}']
            param_names = [x.format(layer) for x in param_names]

            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)

        self.reset_stats()

    def reset_stats(self):
        setattr(self, 'dict_stats', {})
        self.dict_stats['sparsity_delta_x'] = 0
        self.dict_stats['sparsity_delta_h'] = 0
        self.dict_stats['sparsity_delta'] = 0
        self.dict_stats['all_delta_x'] = np.zeros(self.num_layers)
        self.dict_stats['all_delta_h'] = np.zeros(self.num_layers)

    def deltalstm_layer_forward(self, input: Tensor, state: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
                                weight_ih: Parameter, weight_hh: Parameter, i_layer: int, quantize: int = 0,
                                feature_lengths=None):

        # Get Feature Dimension
        dim_feat = input.size(-1)

        # Quantize threshold
        th_x = util.quantize_tensor(torch.tensor(self.th_x, dtype=input.dtype), self.aqi, self.aqf, quantize)
        th_h = util.quantize_tensor(torch.tensor(self.th_h, dtype=input.dtype), self.aqi, self.aqf, quantize)

        # Get Layer Inputs
        inputs = util.quantize_tensor(input, self.aqi, self.aqf, quantize)
        inputs = inputs.unbind(0)

        # Annotate Layer Outputs
        outputs = []

        # Iterate through time steps
        x_prev = util.quantize_tensor(state[0][:, :dim_feat], self.aqi, self.aqf, quantize)
        x_prev_out_size = torch.zeros_like(state[0])
        x_prev_out = util.quantize_tensor(x_prev_out_size, self.aqi, self.aqf, quantize)
        h = util.quantize_tensor(state[1], self.aqi, self.aqf, quantize)
        h_prev = util.quantize_tensor(state[2], self.aqi, self.aqf, quantize)
        c = util.quantize_tensor(state[3], self.aqi, self.aqf, quantize)
        m = state[4]
        l1_norm_delta_h = torch.zeros(1, dtype=input.dtype)  # Intialize L1 Norm of delta h

        # Collect delta vectors for stats
        all_delta_x = []
        all_delta_h = []

        # Iterate through timesteps
        seq_len = len(inputs)
        for i in range(seq_len):
            # Get current input vectors
            x = inputs[i]

            # Get Delta Vectors
            delta_x = x - x_prev
            delta_h = h - h_prev

            # Zero-out elements of delta vector below the threshold
            delta_x_abs = torch.abs(delta_x)
            delta_x = delta_x.masked_fill(delta_x_abs < th_x, 0)
            delta_h_abs = torch.abs(delta_h)
            delta_h = delta_h.masked_fill(delta_h_abs < th_h, 0)

            if not self.training:
                # delta_x = delta_v[:, :dim_feat]
                all_delta_x.append(delta_x.detach().cpu())
                all_delta_h.append(delta_h.detach().cpu())

            # Update previous state vectors memory on indices that had above-threshold change
            x_prev = torch.where(delta_x_abs >= self.th_x, x, x_prev)
            x_prev_out[:, :input.size(-1)] = x_prev
            h_prev = torch.where(delta_h_abs >= self.th_h, h, h_prev)

            # Get l1 norm of delta_h
            l1_norm_delta_h += torch.sum(torch.abs(delta_h.cpu()))

            # Run forward pass for one time step
            m = (torch.mm(delta_x, weight_ih.t()) + torch.mm(delta_h, weight_hh.t())) + m
            pre_act = util.quantize_tensor(m, self.aqi, self.aqf, quantize)
            pre_act_i, pre_act_f, pre_act_g, pre_act_o = pre_act.chunk(4, 1)

            # Compute gates
            if self.hardsigmoid:
                gate_i = hsigmoid(pre_act_i)
                gate_f = hsigmoid(pre_act_f)
                gate_o = hsigmoid(pre_act_o)
            else:
                gate_i = torch.sigmoid(pre_act_i)
                gate_f = torch.sigmoid(pre_act_f)
                gate_o = torch.sigmoid(pre_act_o)
            if self.hardtanh:
                gate_g = F.hardtanh(pre_act_g)
            else:
                gate_g = torch.tanh(pre_act_g)

            q_i = util.quantize_tensor(gate_i, self.nqi, self.nqf, quantize)
            q_f = util.quantize_tensor(gate_f, self.nqi, self.nqf, quantize)
            q_g = util.quantize_tensor(gate_g, self.nqi, self.nqf, quantize)
            q_o = util.quantize_tensor(gate_o, self.nqi, self.nqf, quantize)

            # Compute candidate memory
            mul_cf = torch.mul(c, q_f)
            mul_ig = torch.mul(q_i, q_g)
            c = torch.add(mul_cf, mul_ig)
            q_c = util.quantize_tensor(c, self.aqi, self.aqf, quantize)
            if self.hardtanh:
                c_tanh = F.hardtanh(q_c)
            else:
                c_tanh = torch.tanh(q_c)
            c_tanh = util.quantize_tensor(c_tanh, self.nqi, self.nqf, quantize)
            c = q_c  # Remove

            # Compute next hidden output
            h = torch.mul(q_o, c_tanh)
            h = util.quantize_tensor(h, self.aqi, self.aqf, quantize)

            # Append current DeltaLSTM hidden output to the list
            outputs += [h]

        # Eval stats
        if not self.training:
            all_delta_x = torch.stack(all_delta_x)
            all_delta_h = torch.stack(all_delta_h)
            self.update_stats(all_delta_x, all_delta_h, i_layer, feature_lengths)

        del all_delta_x, all_delta_h
        return torch.stack(outputs), (x_prev_out, h, h_prev, c, m), l1_norm_delta_h

    def initialize_state(self, input: Tensor, state: Optional[Tuple] = None):
        x = input
        # Reshape input if necessary
        if self.batch_first:
            x.transpose(0, 1)
            setattr(self, 'batch_size', int(x.size()[0]))
        else:
            setattr(self, 'batch_size', int(x.size()[1]))
        batch_size = x.size()[1]
        max_seq_len = x.size()[0]

        # Adapt to Mode
        init_input = torch.zeros(batch_size, max(self.input_size, self.hidden_size),
                                 dtype=input.dtype, device=input.device)
        init_state = torch.zeros(batch_size, self.hidden_size,
                                 dtype=input.dtype, device=input.device)
        init_gate = torch.zeros(batch_size, self.gate_size,
                                dtype=input.dtype, device=input.device)
        if self.mode == 'LSTM':
            if state is None:
                state = [(init_state, init_state) for i in range(self.num_layers)]
            else:
                state = [(state[0][i], state[1][i]) for i in range(self.num_layers)]
        elif self.mode == 'DeltaLSTM':
            if state is None:
                list_state = []
                for i_layer in range(self.num_layers):
                    # b = getattr(self, 'bias_l{}'.format(i_layer))
                    # bias_hh = getattr(self, 'bias_hh_l{}'.format(i_layer))
                    # init_m = init_gate + b
                    bias_ih = getattr(self, 'bias_ih_l{}'.format(i_layer))
                    bias_hh = getattr(self, 'bias_hh_l{}'.format(i_layer))
                    init_m = init_gate + bias_ih + bias_hh
                    list_state.append((init_input, init_state, init_state, init_state, init_m))
            else:
                list_state = []
                for i_layer in range(self.num_layers):
                    init_m = state[i_layer][4]
                    list_state.append((state[i_layer][0], state[i_layer][1], state[i_layer][2],
                                       state[i_layer][3], init_m))
            state = list_state

        else:
            raise ValueError("Unrecognized RNN mode: " + self.mode)

        return x, state

    def forward(self, input: Tensor, state: Optional[Tuple] = None, quantize: int = 0, feature_lengths=None):
        # Initialize State
        x, state = self.initialize_state(input, state)

        # Select Forward Propagation Function
        if self.mode == 'LSTM':
            _layer_forward = self.lstm_layer_forward
        elif self.mode == 'DeltaLSTM':
            _layer_forward = self.deltalstm_layer_forward
        else:
            raise ValueError("Unrecognized RNN mode: " + self.mode)

        # Iterate through layers
        reg = torch.zeros(1, dtype=input.dtype)
        layer_state = []
        for i_layer in range(self.num_layers):
            # Get Layer Parameters
            weight_ih = getattr(self, 'weight_ih_l{}'.format(i_layer))
            weight_hh = getattr(self, 'weight_hh_l{}'.format(i_layer))
            # Forward Propation of Layer
            x, i_state, i_reg = _layer_forward(x, state[i_layer], weight_ih, weight_hh, i_layer, quantize,
                                               feature_lengths)
            layer_state += [list(i_state)]
            reg += i_reg

        # Concat layer states
        state = tuple([torch.stack([layer_state[i_layer][i_state] for i_layer in range(self.num_layers)])
                       for i_state in range(self.num_states)])

        # Get Statistics
        if not self.training:
            self.get_temporal_sparsity()

        return x, state, reg

    def adapt_net_to_hw(self):
        # Generate parameter zero pads for hardware adaptation
        for i_layer in range(self.num_layers):
            # Get weights
            weight_ih = getattr(self, 'weight_ih_l{}'.format(i_layer))
            weight_hh = getattr(self, 'weight_hh_l{}'.format(i_layer))
            # Generate Zero Pads
            if i_layer == 0:
                v_pad_ih = torch.zeros((self.diff_hidden_size, self.input_size), device=weight_ih.device)
                h_pad_ih = torch.zeros((self.num_gates * self.adapted_hidden_size, self.diff_input_size),
                                       device=weight_ih.device)
                v_pad_hh = torch.zeros((self.diff_hidden_size, self.hidden_size), device=weight_ih.device)
                h_pad_hh = torch.zeros((self.num_gates * self.adapted_hidden_size, self.diff_hidden_size),
                                       device=weight_ih.device)
            else:
                v_pad_ih = torch.zeros((self.diff_hidden_size, self.hidden_size), device=weight_ih.device)
                h_pad_ih = torch.zeros((self.num_gates * self.adapted_hidden_size, self.diff_hidden_size),
                                       device=weight_ih.device)
                v_pad_hh = torch.zeros((self.diff_hidden_size, self.hidden_size), device=weight_ih.device)
                h_pad_hh = torch.zeros((self.num_gates * self.adapted_hidden_size, self.diff_hidden_size),
                                       device=weight_ih.device)
            # Pad weight matrices
            weight_ih_adapt = list(weight_ih.chunk(self.num_gates, dim=0))
            weight_hh_adapt = list(weight_hh.chunk(self.num_gates, dim=0))
            for i in range(self.num_gates):
                weight_ih_adapt[i] = torch.vstack((weight_ih_adapt[i], v_pad_ih))
                weight_hh_adapt[i] = torch.vstack((weight_hh_adapt[i], v_pad_hh))
            weight_ih_adapt = torch.vstack(weight_ih_adapt)
            weight_ih_adapt = torch.hstack((weight_ih_adapt, h_pad_ih))
            weight_hh_adapt = torch.vstack(weight_hh_adapt)
            weight_hh_adapt = torch.hstack((weight_hh_adapt, h_pad_hh))

            # Update weights with adapted versions
            for name, param in self.named_parameters():
                if name == 'weight_ih_l{}'.format(i_layer):
                    param.data = weight_ih_adapt
                if name == 'weight_hh_l{}'.format(i_layer):
                    param.data = weight_hh_adapt

    def update_stats(self, all_delta_x: Tensor, all_delta_h: Tensor, i_layer: int, feature_lengths=None):
        all_delta_x = all_delta_x.cpu().numpy()
        all_delta_h = all_delta_h.cpu().numpy()
        batch_size = all_delta_x.shape[1]

        # Adapt state sizes to hardware supported size
        inp_size = float(all_delta_x.shape[-1])
        hid_size = float(all_delta_h.shape[-1])
        seq_len = all_delta_x.shape[0]
        adapted_inp_size = int(math.ceil(inp_size / self.num_array) * self.num_array)
        adapted_hid_size = int(math.ceil(hid_size / self.num_array) * self.num_array)
        inp_zero_pad_size = int(adapted_inp_size - inp_size)
        hid_zero_pad_size = int(adapted_hid_size - hid_size)
        all_delta_x_padded = np.concatenate((all_delta_x, np.zeros((seq_len, batch_size, inp_zero_pad_size))), axis=-1)
        all_delta_h_padded = np.concatenate((all_delta_h, np.zeros((seq_len, batch_size, hid_zero_pad_size))), axis=-1)
        all_delta_v = np.concatenate((all_delta_x_padded, all_delta_h_padded), axis=-1)

        ################
        # Get Stats
        ################
        n_zeros_delta_x = 0
        n_zeros_delta_h = 0
        n_zeros_delta_v = 0
        n_elems_delta_x = 0
        n_elems_delta_h = 0
        n_elems_delta_v = 0
        # state_work_delta_x = np.zeros(all_delta_x.shape[-1])
        # state_work_delta_h = np.zeros(all_delta_h.shape[-1])
        state_work = np.zeros(all_delta_v.shape[-1])
        total_feature_length = np.sum(feature_lengths.cpu().numpy())
        num_states_per_array_delta_x = adapted_inp_size / self.num_array
        num_states_per_array_delta_h = adapted_hid_size / self.num_array
        num_states_per_array = (adapted_inp_size + adapted_hid_size) / self.num_array
        state_work_per_array_delta_x = np.zeros((self.num_array, int(num_states_per_array_delta_x)))
        state_work_per_array_delta_h = np.zeros((self.num_array, int(num_states_per_array_delta_h)))
        state_work_per_array = np.zeros((self.num_array, int(num_states_per_array)))
        array_work_delta_x = np.zeros(self.num_array)
        array_work_delta_h = np.zeros(self.num_array)
        array_work = np.zeros(self.num_array)
        mean_array_work = 0.0
        worst_array_work = 0.0
        # Iterate throughput samples
        for i, length in enumerate(feature_lengths):
            delta_x = all_delta_x[:length, i, :]
            delta_x_padded = all_delta_x_padded[:length, i, :]
            delta_h = all_delta_h[:length, i, :]
            delta_h_padded = all_delta_h_padded[:length, i, :]
            delta_v = all_delta_v[:length, i, :]
            zero_mask_delta_x = np.asarray(delta_x == 0, dtype=np.float32)
            zero_mask_delta_h = np.asarray(delta_h == 0, dtype=np.float32)
            zero_mask_delta_v = np.asarray(delta_v == 0, dtype=np.float32)
            nonzero_mask_delta_x_padded = np.asarray(delta_x_padded != 0, dtype=np.float32)
            nonzero_mask_delta_h_padded = np.asarray(delta_h_padded != 0, dtype=np.float32)
            nonzero_mask_delta_v = np.asarray(delta_v != 0, dtype=np.float32)

            # Number of zeros in delta vectors
            n_zeros_delta_x += np.sum(zero_mask_delta_x)
            n_zeros_delta_h += np.sum(zero_mask_delta_h)
            n_zeros_delta_v += np.sum(zero_mask_delta_v)

            # Number of elements in delta vectors
            n_elems_delta_x += float(delta_x.size)
            n_elems_delta_h += float(delta_h.size)
            n_elems_delta_v += float(delta_v.size)

            # Get state work of the current sample (T, F)
            state_work_delta_x_t = nonzero_mask_delta_x_padded
            state_work_delta_h_t = nonzero_mask_delta_h_padded
            state_work_t = nonzero_mask_delta_v

            # State workload of delta vectors (F)
            state_work += np.sum(state_work_t, axis=0)

            # Get state workload per array for all timesteps (T, num_array, num_states_per_array)
            state_work_per_array_delta_x_t = np.zeros((length, self.num_array, int(num_states_per_array_delta_x)))
            state_work_per_array_delta_h_t = np.zeros((length, self.num_array, int(num_states_per_array_delta_h)))
            state_work_per_array_t = np.zeros((length, self.num_array, int(num_states_per_array)))
            for j in range(self.num_array):
                state_work_per_array_delta_x_t[:, j, :] = state_work_delta_x_t[:, j::self.num_array]
                state_work_per_array_delta_h_t[:, j, :] = state_work_delta_h_t[:, j::self.num_array]
                state_work_per_array_t[:, j, :] = state_work_t[:, j::self.num_array]

            # Get state workload per array (num_array, num_states_per_array)
            state_work_per_array_delta_x += np.sum(state_work_per_array_delta_x_t, axis=0)
            state_work_per_array_delta_h += np.sum(state_work_per_array_delta_h_t, axis=0)
            state_work_per_array += np.sum(state_work_per_array_t, axis=0)

            # Get array work for all timesteps (T, num_array)
            array_work_delta_x_t = np.sum(state_work_per_array_delta_x_t, axis=-1)
            array_work_delta_h_t = np.sum(state_work_per_array_delta_h_t, axis=-1)
            array_work_t = np.sum(state_work_per_array_t, axis=-1)

            # Get array workload (num_array)
            array_work_delta_x += np.sum(array_work_delta_x_t, axis=0)
            array_work_delta_h += np.sum(array_work_delta_h_t, axis=0)
            array_work += np.sum(array_work_t, axis=0)

            # Get mean array workload (1)
            mean_array_work += np.sum(np.mean(array_work_t, axis=-1))

            # Get worst array workload (1)
            worst_array_work += np.sum(np.max(array_work_t, axis=-1))

            # Get number of being the worst array
            max_mask = (np.max(array_work_t, axis=-1, keepdims=True) == array_work_t)
            if i == 0:
                num_worst_array = np.sum(max_mask, axis=0, dtype=np.float64)
            else:
                num_worst_array += np.sum(max_mask, axis=0, dtype=np.float64)

        # Evaluate RNN Weight Sparsity
        n_nonzero_weight_elem = 0
        n_weight_elem = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                n_nonzero_weight_elem += len(torch.nonzero(param.data))
                n_weight_elem += param.data.nelement()
        sp_rnn = 1 - (n_nonzero_weight_elem / n_weight_elem)

        # Get performance stats
        worst_total_array_work = worst_array_work * self.num_array
        n_ops_dense = 2 * (inp_size + hid_size) * self.gate_size * total_feature_length

        n_ops_sparse = 2 * worst_total_array_work * self.gate_size * (1 - sp_rnn)

        dict_stats = {
            'seq_len': total_feature_length,
            'n_zeros_delta_x': n_zeros_delta_x,
            'n_zeros_delta_h': n_zeros_delta_h,
            'n_zeros_delta_v': n_zeros_delta_v,
            'n_elems_delta_x': n_elems_delta_x,
            'n_elems_delta_h': n_elems_delta_h,
            'n_elems_delta_v': n_elems_delta_v,
            'state_work_l' + str(i_layer): state_work,
            'state_work_per_array_l' + str(i_layer): state_work_per_array,
            'state_work_per_array_delta_x_l' + str(i_layer): state_work_per_array_delta_x,
            'state_work_per_array_delta_h_l' + str(i_layer): state_work_per_array_delta_h,
            'array_work_delta_x_l' + str(i_layer): array_work_delta_x,
            'array_work_delta_h_l' + str(i_layer): array_work_delta_h,
            'array_work_l' + str(i_layer): array_work,
            'mean_array_work_l' + str(i_layer): mean_array_work,
            'worst_array_work_l' + str(i_layer): worst_array_work,
            'num_worst_array_l' + str(i_layer): num_worst_array,
            'n_ops_dense_l' + str(i_layer): n_ops_dense,
            'n_ops_sparse_l' + str(i_layer): n_ops_sparse,
        }

        for k, v in dict_stats.items():
            if k in self.dict_stats.keys():
                self.dict_stats[k] += v
            else:
                self.dict_stats[k] = v

        return self.dict_stats

    def get_temporal_sparsity(self):
        if not self.training:
            # Get stats
            n_zeros_delta_x = self.dict_stats['n_zeros_delta_x']
            n_zeros_delta_h = self.dict_stats['n_zeros_delta_h']
            n_elems_delta_x = self.dict_stats['n_elems_delta_x']
            n_elems_delta_h = self.dict_stats['n_elems_delta_h']
            # Calculate sparsity
            sparsity_delta_x = n_zeros_delta_x / n_elems_delta_x
            sparsity_delta_h = n_zeros_delta_h / n_elems_delta_h
            sparsity_delta = (n_zeros_delta_x + n_zeros_delta_h) / (n_elems_delta_x + n_elems_delta_h)
            self.dict_stats['sparsity_delta_x'] = sparsity_delta_x
            self.dict_stats['sparsity_delta_h'] = sparsity_delta_h
            self.dict_stats['sparsity_delta'] = sparsity_delta
            return self.dict_stats
        else:
            raise RuntimeError('DeltaRNN statistics can only be updated during evaluation phase.')

    def get_workload(self):
        if not self.training:
            # Get total seq len
            seq_len = float(self.dict_stats['seq_len'])

            # Iterate through layers
            expect_mean_array_work = self.dict_stats['mean_array_work_l0']
            expect_worst_array_work = self.dict_stats['worst_array_work_l0']
            array_work = self.dict_stats['array_work_l0']
            n_ops_dense = self.dict_stats['n_ops_dense_l0']
            n_ops_sparse = self.dict_stats['n_ops_sparse_l0']
            self.dict_stats['prop_state_work_per_array_l0'] \
                = self.dict_stats['state_work_per_array_l0'] / seq_len
            self.dict_stats['prop_state_work_per_array_delta_x_l0'] \
                = self.dict_stats['state_work_per_array_delta_x_l0'] / seq_len
            self.dict_stats['prop_state_work_per_array_delta_h_l0'] \
                = self.dict_stats['state_work_per_array_delta_h_l0'] / seq_len
            self.dict_stats['prop_array_work_delta_x_l0'] \
                = self.dict_stats['array_work_delta_x_l0'] / seq_len
            self.dict_stats['prop_array_work_delta_h_l0'] \
                = self.dict_stats['array_work_delta_h_l0'] / seq_len
            for i in range(1, self.num_layers):
                expect_mean_array_work += self.dict_stats['mean_array_work_l' + str(i)]
                expect_worst_array_work += self.dict_stats['worst_array_work_l' + str(i)]
                array_work += self.dict_stats['array_work_l' + str(i)]
                n_ops_dense += self.dict_stats['n_ops_dense_l' + str(i)]
                n_ops_sparse += self.dict_stats['n_ops_sparse_l' + str(i)]
                self.dict_stats['prop_state_work_l' + str(i)] = self.dict_stats['state_work_l' + str(i)] / seq_len
                self.dict_stats['prop_state_work_per_array_delta_x_l' + str(i)] \
                    = self.dict_stats['state_work_per_array_delta_x_l' + str(i)] / seq_len
                self.dict_stats['prop_state_work_per_array_delta_h_l' + str(i)] \
                    = self.dict_stats['state_work_per_array_delta_h_l' + str(i)] / seq_len
                self.dict_stats['prop_state_work_per_array_l' + str(i)] \
                    = self.dict_stats['state_work_per_array_l' + str(i)] / seq_len
                self.dict_stats['prop_array_work_delta_x_l' + str(i)] \
                    = self.dict_stats['array_work_delta_x_l' + str(i)] / seq_len
                self.dict_stats['prop_array_work_delta_h_l' + str(i)] \
                    = self.dict_stats['array_work_delta_h_l' + str(i)] / seq_len
                self.dict_stats['prop_array_work_l' + str(i)] \
                    = self.dict_stats['array_work_l' + str(i)] / seq_len

            # Expectation of array work
            expect_array_work = array_work / float(seq_len * self.num_layers)
            expect_mean_array_work = expect_mean_array_work / float(seq_len * self.num_layers)
            expect_worst_array_work = expect_worst_array_work / float(seq_len * self.num_layers)
            imbalance = float(expect_worst_array_work) / float(expect_mean_array_work)
            balance = float(expect_mean_array_work) / float(expect_worst_array_work)
            self.dict_stats['expect_array_work'] = expect_array_work
            self.dict_stats['expect_worst_array_work'] = expect_worst_array_work
            self.dict_stats['expect_mean_array_work'] = expect_mean_array_work
            self.dict_stats['imbalance'] = imbalance
            self.dict_stats['balance'] = balance
            return self.dict_stats
        else:
            raise RuntimeError('DeltaRNN statistics can only be updated during evaluation phase.')

    def gen_init_state(self, batch_size, input_size, hidden_size, gate_size, num_layers, use_cuda):
        init_input = torch.zeros(batch_size, max(input_size, hidden_size)).float()
        init_state = torch.zeros(batch_size, hidden_size).float()
        # init_gate = torch.zeros(batch_size, gate_size).float()
        x_prev_0, h_0, h_prev_0, c_0, m_0 = [], [], [], [], []

        for i in range(num_layers):
            x_prev_0.append(init_input)
            h_0.append(init_state)
            h_prev_0.append(init_state)
            c_0.append(init_state)
            bias_ih = getattr(self, 'bias_ih_l{}'.format(i))
            bias_hh = getattr(self, 'bias_hh_l{}'.format(i))
            m_0.append(bias_ih + bias_hh)
        x_prev_0 = torch.stack(x_prev_0)
        h_0 = torch.stack(h_0)
        h_prev_0 = torch.stack(h_prev_0)
        c_0 = torch.stack(c_0)
        m_0 = torch.stack(m_0)
        if use_cuda:
            x_prev_0 = x_prev_0.cuda()
            h_0 = h_0.cuda()
            h_prev_0 = h_prev_0.cuda()
            c_0 = c_0.cuda()
            m_0 = m_0.cuda()
        return x_prev_0, h_0, h_prev_0, c_0, m_0

    def balance_workload(self):
        # Clear hidden swap list
        self.hidden_swap_list = []
        for i in range(0, self.num_layers):
            self.hidden_swap_list.append(np.arange(0, self.adapted_hidden_size))

        # Get necessary workload stats
        gs_wl_delta_x \
            = [self.dict_stats['prop_state_work_per_array_delta_x_l' + str(i)] for i in range(self.num_layers)]
        gs_wl_delta_h \
            = [self.dict_stats['prop_state_work_per_array_delta_h_l' + str(i)] for i in range(self.num_layers)]

        # Select stats
        list_gs_wl = [gs_wl_delta_x[0]]  # List of state work
        for i in range(0, self.num_layers):
            list_gs_wl.append(gs_wl_delta_h[i])

        # Create swap lists
        list_swap_lists = [np.arange(0, self.adapted_input_size, dtype=int)]
        for i in range(0, self.num_layers):
            list_swap_lists.append(np.arange(0, self.adapted_hidden_size, dtype=int))

        # Balance workload for each layer (input layer + hidden layers)
        for l in range(len(list_gs_wl)):
            # Get stats for the current layer
            layer_gs_wl = list_gs_wl[l]  # Grouped state workload of the current layer
            layer_g_wl = np.sum(layer_gs_wl, axis=-1)  # Group workload of the current layer
            layer_swap_list = list_swap_lists[l]  # Swap list of the current layer
            layer_mean_g_wl = np.mean(layer_g_wl)

            # Sort group workload
            _, rank_layer_g_wl_des = torch.sort(torch.from_numpy(layer_g_wl), descending=True)
            rank_layer_g_wl_des = rank_layer_g_wl_des.numpy()
            _, rank_layer_g_wl_asc = torch.sort(torch.from_numpy(layer_g_wl), descending=False)
            rank_layer_g_wl_asc = rank_layer_g_wl_asc.numpy()

            # Iterate through Arrays from High Rank to Low Rank
            # in order to Swap Arrays with Rank r and r+1 (Rank 0 is the highest)
            for i in range(0, self.num_array - 1):  # Higher Rank Array
                layer_g_wl = np.sum(layer_gs_wl, axis=-1)  # Group workload of the current layer
                layer_worst_g_wl = np.max(layer_g_wl)
                layer_balance = layer_mean_g_wl / layer_worst_g_wl
                # print(layer_balance)
                # Get index of the higher ranked group
                hg_idx = rank_layer_g_wl_des[i]
                # for j in range(i+i, self.num_array):  # Lower Rank Array
                for j in range(0, self.num_array - i - 1):  # Lower Rank Array
                    # Get index of lower groups
                    lg_idx = rank_layer_g_wl_asc[j]
                    # Get group workload of the higher ranked group
                    hg_wl = np.sum(layer_gs_wl[hg_idx])
                    # Jump to next higher group if the current higher group workload is already lower than the mean
                    if hg_wl <= layer_mean_g_wl:
                        break
                    # Sort grouped state workload
                    sorted_layer_gs_wl_asc, rank_layer_gs_wl_asc = torch.sort(torch.from_numpy(layer_gs_wl), 
                                                                              descending=False, dim=-1)
                    sorted_layer_gs_wl_asc, rank_layer_gs_wl_asc = \
                        sorted_layer_gs_wl_asc.numpy(), rank_layer_gs_wl_asc.numpy()
                    sorted_layer_gs_wl_dec, rank_layer_gs_wl_dec = torch.sort(torch.from_numpy(layer_gs_wl),
                                                                              descending=True, dim=-1)
                    sorted_layer_gs_wl_dec, rank_layer_gs_wl_dec = \
                        sorted_layer_gs_wl_dec.numpy(), rank_layer_gs_wl_dec.numpy()
                    # Iterate through grouped states
                    num_grouped_states = layer_gs_wl.shape[-1]
                    for k in range(num_grouped_states):
                        # for k in range(num_grouped_states):
                        if k >= num_grouped_states / 2:
                            break
                        # Evaluate group workload of higher & lower rank arrays
                        hg_wl = np.sum(layer_gs_wl[hg_idx])
                        lg_wl = np.sum(layer_gs_wl[lg_idx])
                        # If higher ranked group workload is smaller or equal to
                        # the mean group workload and vice versa for the lower
                        # ranked group workload, break the swap loop
                        if hg_wl <= layer_mean_g_wl or lg_wl >= layer_mean_g_wl:
                            break
                        # Otherwise try to swap the states
                        else:
                            # Get index of higher and lower ranked grouped states
                            hgs_idx = rank_layer_gs_wl_asc[hg_idx, k]
                            lgs_idx = rank_layer_gs_wl_asc[lg_idx, k]
                            # Swap group workload
                            hgs_wl = layer_gs_wl[hg_idx, hgs_idx]
                            lgs_wl = layer_gs_wl[lg_idx, lgs_idx]
                            diff_gs_wl = hgs_wl - lgs_wl
                            if diff_gs_wl <= 0:  # Skip to the next grouped state
                                continue
                            else:  # Do the actual swap
                                layer_gs_wl[hg_idx, hgs_idx], layer_gs_wl[lg_idx, lgs_idx] = \
                                    layer_gs_wl[lg_idx, lgs_idx], layer_gs_wl[hg_idx, hgs_idx]  # Swap array workload
                                # Update swap list
                                hs_idx = hgs_idx * self.num_array + hg_idx  # Higher state index
                                ls_idx = lgs_idx * self.num_array + lg_idx  # Lower state index
                                layer_swap_list[hs_idx], layer_swap_list[ls_idx] = \
                                    layer_swap_list[ls_idx], layer_swap_list[hs_idx]  # Swap state index
            # Update swap list
            list_swap_lists[l] = layer_swap_list
        # Update network input swap list
        self.input_swap_list = self.input_swap_list[list_swap_lists[0]]
        for l in range(self.num_layers):
            self.hidden_swap_list[l] = list_swap_lists[l+1]
        # Swap rows and columns of weights
        for l in range(0, self.num_layers):
            input_swap_list = list_swap_lists[l]
            hidden_swap_list = list_swap_lists[l+1]
            for name, param in self.named_parameters():
                if name == 'bias_ih_l{}'.format(l) or name == 'bias_hh_l{}'.format(l):
                    print("::: Swap Parameter: ", name)
                    bias = param.data
                    # Swap rows
                    gates = torch.chunk(bias, self.num_gates, dim=0)
                    gates = [gate[hidden_swap_list] for gate in gates]
                    bias = torch.cat(gates)
                    # Update parameter
                    param.data = bias
                if name == 'weight_ih_l{}'.format(l):
                    print("::: Swap Parameter: ", name)
                    weight = param.data
                    # Swap cols
                    weight = weight[:, input_swap_list]
                    # Swap rows
                    gates = torch.chunk(weight, self.num_gates, dim=0)
                    gates = [gate[hidden_swap_list, :] for gate in gates]
                    weight = torch.vstack(gates)
                    # Update parameter
                    param.data = weight
                if name == 'weight_hh_l{}'.format(l):
                    print("::: Swap Parameter: ", name)
                    weight = param.data
                    # Swap cols
                    weight = weight[:, hidden_swap_list]
                    # Swap rows
                    gates = torch.chunk(weight, self.num_gates, dim=0)
                    gates = [gate[hidden_swap_list, :] for gate in gates]
                    weight = torch.vstack(gates)
                    # Update parameter
                    param.data = weight
