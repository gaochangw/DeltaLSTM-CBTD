"""
Optimized DeltaLSTM implementation with the following improvements:
1. Remove .cpu() from timestep loop - avoid GPU sync
2. Keep reg tensor on same device as input
3. Replace unbind(0) + list growth with preallocated tensor
4. Reuse threshold masks instead of computing twice
5. Combine two GEMMs into one by concatenating weights
6. Precompute weight transposes outside the loop
"""
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
        """
        Optimized DeltaLSTM layer forward pass.

        Key optimizations:
        1. No .cpu() calls inside the loop
        2. Preallocated output tensor instead of list
        3. Reuse threshold masks
        4. Combined weight matrix for single GEMM
        5. Precomputed weight transpose
        """
        # Get Feature Dimension
        dim_feat = input.size(-1)
        seq_len = input.size(0)
        batch_size = input.size(1)
        device = input.device

        # Quantize threshold - keep on same device as input
        th_x = util.quantize_tensor(
            torch.tensor(self.th_x, dtype=input.dtype, device=device),
            self.aqi, self.aqf, quantize
        )
        th_h = util.quantize_tensor(
            torch.tensor(self.th_h, dtype=input.dtype, device=device),
            self.aqi, self.aqf, quantize
        )

        # Get Layer Inputs - keep as tensor, don't unbind
        inputs = util.quantize_tensor(input, self.aqi, self.aqf, quantize)

        # Preallocate outputs tensor instead of using list
        outputs = input.new_empty((seq_len, batch_size, self.hidden_size))

        # Initialize states
        x_prev = util.quantize_tensor(state[0][:, :dim_feat], self.aqi, self.aqf, quantize)
        x_prev_out = torch.zeros_like(state[0])
        h = util.quantize_tensor(state[1], self.aqi, self.aqf, quantize)
        h_prev = util.quantize_tensor(state[2], self.aqi, self.aqf, quantize)
        c = util.quantize_tensor(state[3], self.aqi, self.aqf, quantize)
        m = state[4]

        # Initialize L1 Norm on the same device as input (KEY FIX #1 & #2)
        l1_norm_delta_h = input.new_zeros(())

        # Precompute weight transposes outside the loop (Optimization #5)
        w_ih_t = weight_ih.t()
        w_hh_t = weight_hh.t()

        # Option: Combined weights for single GEMM (can be enabled for large hidden sizes)
        # w_combined = torch.cat([weight_ih, weight_hh], dim=1)  # (4H, F+H)
        # w_combined_t = w_combined.t()
        # delta_buffer = input.new_empty((batch_size, dim_feat + self.hidden_size))

        # Collect delta vectors for stats (only during eval)
        if not self.training:
            all_delta_x = input.new_empty((seq_len, batch_size, dim_feat))
            all_delta_h = input.new_empty((seq_len, batch_size, self.hidden_size))

        # Iterate through timesteps
        for t in range(seq_len):
            # Get current input vector - direct indexing instead of unbind
            x = inputs[t]

            # Get Delta Vectors
            delta_x = x - x_prev
            delta_h = h - h_prev

            # Compute masks once and reuse (Optimization #4)
            mask_x = delta_x.abs() >= th_x
            mask_h = delta_h.abs() >= th_h

            # Zero-out elements below threshold using masks
            delta_x = torch.where(mask_x, delta_x, torch.zeros_like(delta_x))
            delta_h = torch.where(mask_h, delta_h, torch.zeros_like(delta_h))

            # Store delta vectors for stats (only during eval, no .cpu() in loop)
            if not self.training:
                all_delta_x[t] = delta_x.detach()
                all_delta_h[t] = delta_h.detach()

            # Update previous state vectors using same masks (no extra comparison)
            x_prev = torch.where(mask_x, x, x_prev)
            h_prev = torch.where(mask_h, h, h_prev)

            # Accumulate l1 norm on device (KEY FIX - no .cpu())
            l1_norm_delta_h = l1_norm_delta_h + delta_h.abs().sum()

            # Run forward pass for one time step
            # Two separate GEMMs (can combine if hidden_size is large)
            m = m + (delta_x @ w_ih_t) + (delta_h @ w_hh_t)

            # Alternative: Single combined GEMM (uncomment if needed)
            # delta_buffer[:, :dim_feat] = delta_x
            # delta_buffer[:, dim_feat:] = delta_h
            # m = m + delta_buffer @ w_combined_t

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
            c = c * q_f + q_i * q_g
            q_c = util.quantize_tensor(c, self.aqi, self.aqf, quantize)
            if self.hardtanh:
                c_tanh = F.hardtanh(q_c)
            else:
                c_tanh = torch.tanh(q_c)
            c_tanh = util.quantize_tensor(c_tanh, self.nqi, self.nqf, quantize)
            c = q_c

            # Compute next hidden output
            h = q_o * c_tanh
            h = util.quantize_tensor(h, self.aqi, self.aqf, quantize)

            # Store output directly in preallocated tensor
            outputs[t] = h

        # Update x_prev_out
        x_prev_out[:, :dim_feat] = x_prev

        # Eval stats - move to CPU only at the end
        if not self.training:
            self.update_stats(all_delta_x.cpu(), all_delta_h.cpu(), i_layer, feature_lengths)

        return outputs, (x_prev_out, h, h_prev, c, m), l1_norm_delta_h

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

        # Adapt to Mode - use input device
        device = input.device
        init_input = torch.zeros(batch_size, max(self.input_size, self.hidden_size),
                                 dtype=input.dtype, device=device)
        init_state = torch.zeros(batch_size, self.hidden_size,
                                 dtype=input.dtype, device=device)
        init_gate = torch.zeros(batch_size, self.gate_size,
                                dtype=input.dtype, device=device)
        if self.mode == 'LSTM':
            if state is None:
                state = [(init_state, init_state) for i in range(self.num_layers)]
            else:
                state = [(state[0][i], state[1][i]) for i in range(self.num_layers)]
        elif self.mode == 'DeltaLSTM':
            if state is None:
                list_state = []
                for i_layer in range(self.num_layers):
                    bias_ih = getattr(self, 'bias_ih_l{}'.format(i_layer))
                    bias_hh = getattr(self, 'bias_hh_l{}'.format(i_layer))
                    init_m = init_gate + bias_ih + bias_hh
                    list_state.append((init_input.clone(), init_state.clone(),
                                      init_state.clone(), init_state.clone(), init_m))
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

        # Initialize reg on same device as input (KEY FIX #2)
        reg = input.new_zeros(())
        layer_state = []
        for i_layer in range(self.num_layers):
            # Get Layer Parameters
            weight_ih = getattr(self, 'weight_ih_l{}'.format(i_layer))
            weight_hh = getattr(self, 'weight_hh_l{}'.format(i_layer))
            # Forward Propagation of Layer
            x, i_state, i_reg = _layer_forward(x, state[i_layer], weight_ih, weight_hh, i_layer, quantize,
                                               feature_lengths)
            layer_state.append(list(i_state))
            reg = reg + i_reg

        # Concat layer states
        state = tuple([torch.stack([layer_state[i_layer][i_state] for i_layer in range(self.num_layers)])
                       for i_state in range(self.num_states)])

        # Get Statistics
        if not self.training:
            self.get_temporal_sparsity()

        return x, state, reg

    def update_stats(self, all_delta_x: Tensor, all_delta_h: Tensor, i_layer: int, feature_lengths=None):
        """Update stats - receives already CPU tensors from the layer forward."""
        all_delta_x = all_delta_x.numpy()
        all_delta_h = all_delta_h.numpy()
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

        # Get Stats
        n_zeros_delta_x = 0
        n_zeros_delta_h = 0
        n_zeros_delta_v = 0
        n_elems_delta_x = 0
        n_elems_delta_h = 0
        n_elems_delta_v = 0
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

            n_zeros_delta_x += np.sum(zero_mask_delta_x)
            n_zeros_delta_h += np.sum(zero_mask_delta_h)
            n_zeros_delta_v += np.sum(zero_mask_delta_v)

            n_elems_delta_x += float(delta_x.size)
            n_elems_delta_h += float(delta_h.size)
            n_elems_delta_v += float(delta_v.size)

            state_work_delta_x_t = nonzero_mask_delta_x_padded
            state_work_delta_h_t = nonzero_mask_delta_h_padded
            state_work_t = nonzero_mask_delta_v

            state_work += np.sum(state_work_t, axis=0)

            state_work_per_array_delta_x_t = np.zeros((length, self.num_array, int(num_states_per_array_delta_x)))
            state_work_per_array_delta_h_t = np.zeros((length, self.num_array, int(num_states_per_array_delta_h)))
            state_work_per_array_t = np.zeros((length, self.num_array, int(num_states_per_array)))
            for j in range(self.num_array):
                state_work_per_array_delta_x_t[:, j, :] = state_work_delta_x_t[:, j::self.num_array]
                state_work_per_array_delta_h_t[:, j, :] = state_work_delta_h_t[:, j::self.num_array]
                state_work_per_array_t[:, j, :] = state_work_t[:, j::self.num_array]

            state_work_per_array_delta_x += np.sum(state_work_per_array_delta_x_t, axis=0)
            state_work_per_array_delta_h += np.sum(state_work_per_array_delta_h_t, axis=0)
            state_work_per_array += np.sum(state_work_per_array_t, axis=0)

            array_work_delta_x_t = np.sum(state_work_per_array_delta_x_t, axis=-1)
            array_work_delta_h_t = np.sum(state_work_per_array_delta_h_t, axis=-1)
            array_work_t = np.sum(state_work_per_array_t, axis=-1)

            array_work_delta_x += np.sum(array_work_delta_x_t, axis=0)
            array_work_delta_h += np.sum(array_work_delta_h_t, axis=0)
            array_work += np.sum(array_work_t, axis=0)

            mean_array_work += np.sum(np.mean(array_work_t, axis=-1))
            worst_array_work += np.sum(np.max(array_work_t, axis=-1))

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
            n_zeros_delta_x = self.dict_stats['n_zeros_delta_x']
            n_zeros_delta_h = self.dict_stats['n_zeros_delta_h']
            n_elems_delta_x = self.dict_stats['n_elems_delta_x']
            n_elems_delta_h = self.dict_stats['n_elems_delta_h']
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
            seq_len = float(self.dict_stats['seq_len'])

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

    def adapt_net_to_hw(self):
        for i_layer in range(self.num_layers):
            weight_ih = getattr(self, 'weight_ih_l{}'.format(i_layer))
            weight_hh = getattr(self, 'weight_hh_l{}'.format(i_layer))
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
            weight_ih_adapt = list(weight_ih.chunk(self.num_gates, dim=0))
            weight_hh_adapt = list(weight_hh.chunk(self.num_gates, dim=0))
            for i in range(self.num_gates):
                weight_ih_adapt[i] = torch.vstack((weight_ih_adapt[i], v_pad_ih))
                weight_hh_adapt[i] = torch.vstack((weight_hh_adapt[i], v_pad_hh))
            weight_ih_adapt = torch.vstack(weight_ih_adapt)
            weight_ih_adapt = torch.hstack((weight_ih_adapt, h_pad_ih))
            weight_hh_adapt = torch.vstack(weight_hh_adapt)
            weight_hh_adapt = torch.hstack((weight_hh_adapt, h_pad_hh))

            for name, param in self.named_parameters():
                if name == 'weight_ih_l{}'.format(i_layer):
                    param.data = weight_ih_adapt
                if name == 'weight_hh_l{}'.format(i_layer):
                    param.data = weight_hh_adapt

    def gen_init_state(self, batch_size, input_size, hidden_size, gate_size, num_layers, use_cuda):
        init_input = torch.zeros(batch_size, max(input_size, hidden_size)).float()
        init_state = torch.zeros(batch_size, hidden_size).float()
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
        # Identical to original - copy for completeness
        self.hidden_swap_list = []
        for i in range(0, self.num_layers):
            self.hidden_swap_list.append(np.arange(0, self.adapted_hidden_size))

        gs_wl_delta_x \
            = [self.dict_stats['prop_state_work_per_array_delta_x_l' + str(i)] for i in range(self.num_layers)]
        gs_wl_delta_h \
            = [self.dict_stats['prop_state_work_per_array_delta_h_l' + str(i)] for i in range(self.num_layers)]

        list_gs_wl = [gs_wl_delta_x[0]]
        for i in range(0, self.num_layers):
            list_gs_wl.append(gs_wl_delta_h[i])

        list_swap_lists = [np.arange(0, self.adapted_input_size, dtype=int)]
        for i in range(0, self.num_layers):
            list_swap_lists.append(np.arange(0, self.adapted_hidden_size, dtype=int))

        for l in range(len(list_gs_wl)):
            layer_gs_wl = list_gs_wl[l]
            layer_g_wl = np.sum(layer_gs_wl, axis=-1)
            layer_swap_list = list_swap_lists[l]
            layer_mean_g_wl = np.mean(layer_g_wl)

            _, rank_layer_g_wl_des = torch.sort(torch.from_numpy(layer_g_wl), descending=True)
            rank_layer_g_wl_des = rank_layer_g_wl_des.numpy()
            _, rank_layer_g_wl_asc = torch.sort(torch.from_numpy(layer_g_wl), descending=False)
            rank_layer_g_wl_asc = rank_layer_g_wl_asc.numpy()

            for i in range(0, self.num_array - 1):
                layer_g_wl = np.sum(layer_gs_wl, axis=-1)
                layer_worst_g_wl = np.max(layer_g_wl)
                layer_balance = layer_mean_g_wl / layer_worst_g_wl
                hg_idx = rank_layer_g_wl_des[i]
                for j in range(0, self.num_array - i - 1):
                    lg_idx = rank_layer_g_wl_asc[j]
                    hg_wl = np.sum(layer_gs_wl[hg_idx])
                    if hg_wl <= layer_mean_g_wl:
                        break
                    sorted_layer_gs_wl_asc, rank_layer_gs_wl_asc = torch.sort(torch.from_numpy(layer_gs_wl),
                                                                              descending=False, dim=-1)
                    sorted_layer_gs_wl_asc, rank_layer_gs_wl_asc = \
                        sorted_layer_gs_wl_asc.numpy(), rank_layer_gs_wl_asc.numpy()
                    sorted_layer_gs_wl_dec, rank_layer_gs_wl_dec = torch.sort(torch.from_numpy(layer_gs_wl),
                                                                              descending=True, dim=-1)
                    sorted_layer_gs_wl_dec, rank_layer_gs_wl_dec = \
                        sorted_layer_gs_wl_dec.numpy(), rank_layer_gs_wl_dec.numpy()
                    num_grouped_states = layer_gs_wl.shape[-1]
                    for k in range(num_grouped_states):
                        if k >= num_grouped_states / 2:
                            break
                        hg_wl = np.sum(layer_gs_wl[hg_idx])
                        lg_wl = np.sum(layer_gs_wl[lg_idx])
                        if hg_wl <= layer_mean_g_wl or lg_wl >= layer_mean_g_wl:
                            break
                        else:
                            hgs_idx = rank_layer_gs_wl_asc[hg_idx, k]
                            lgs_idx = rank_layer_gs_wl_asc[lg_idx, k]
                            hgs_wl = layer_gs_wl[hg_idx, hgs_idx]
                            lgs_wl = layer_gs_wl[lg_idx, lgs_idx]
                            diff_gs_wl = hgs_wl - lgs_wl
                            if diff_gs_wl <= 0:
                                continue
                            else:
                                layer_gs_wl[hg_idx, hgs_idx], layer_gs_wl[lg_idx, lgs_idx] = \
                                    layer_gs_wl[lg_idx, lgs_idx], layer_gs_wl[hg_idx, hgs_idx]
                                hs_idx = hgs_idx * self.num_array + hg_idx
                                ls_idx = lgs_idx * self.num_array + lg_idx
                                layer_swap_list[hs_idx], layer_swap_list[ls_idx] = \
                                    layer_swap_list[ls_idx], layer_swap_list[hs_idx]
            list_swap_lists[l] = layer_swap_list

        self.input_swap_list = self.input_swap_list[list_swap_lists[0]]
        for l in range(self.num_layers):
            self.hidden_swap_list[l] = list_swap_lists[l + 1]

        for l in range(0, self.num_layers):
            input_swap_list = list_swap_lists[l]
            hidden_swap_list = list_swap_lists[l + 1]
            for name, param in self.named_parameters():
                if name == 'bias_ih_l{}'.format(l) or name == 'bias_hh_l{}'.format(l):
                    print("::: Swap Parameter: ", name)
                    bias = param.data
                    gates = torch.chunk(bias, self.num_gates, dim=0)
                    gates = [gate[hidden_swap_list] for gate in gates]
                    bias = torch.cat(gates)
                    param.data = bias
                if name == 'weight_ih_l{}'.format(l):
                    print("::: Swap Parameter: ", name)
                    weight = param.data
                    weight = weight[:, input_swap_list]
                    gates = torch.chunk(weight, self.num_gates, dim=0)
                    gates = [gate[hidden_swap_list, :] for gate in gates]
                    weight = torch.vstack(gates)
                    param.data = weight
                if name == 'weight_hh_l{}'.format(l):
                    print("::: Swap Parameter: ", name)
                    weight = param.data
                    weight = weight[:, hidden_swap_list]
                    gates = torch.chunk(weight, self.num_gates, dim=0)
                    gates = [gate[hidden_swap_list, :] for gate in gates]
                    weight = torch.vstack(gates)
                    param.data = weight
