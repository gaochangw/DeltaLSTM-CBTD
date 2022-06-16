import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.util as util
from torch.nn.utils.rnn import pack_padded_sequence


def get_batch_data(args, dict_batch_array):
    # Fetch Batch Data
    features = dict_batch_array['features']  # Dim: (T, N, F)
    feature_lengths = dict_batch_array['feature_lengths']
    # features = pack_padded_sequence(features, feature_lengths, enforce_sorted=False)
    if args.phn == 61:
        targets = dict_batch_array['targets_61']
    else:
        targets = dict_batch_array['targets_48']
    targets_metric = dict_batch_array['targets_39']
    target_lengths = dict_batch_array['target_lengths']
    if args.use_cuda:
        features = features.cuda()

    dict_batch_tensor = {'features': features, 'feature_lengths': feature_lengths, 'targets': targets,
                         'targets_metric': targets_metric, 'target_lengths': target_lengths}

    return dict_batch_tensor


def forward_propagation(net, dict_batch_tensor, state=None):
    outputs, state, reg = net(dict_batch_tensor['features'], state, dict_batch_tensor['feature_lengths'])
    out_final, out_rnn = outputs
    return out_final, state, reg


def calculate_loss(loss_fn, net_out, dict_targets, reg, beta):
    loss_fn_input = F.log_softmax(net_out, dim=-1).transpose(0, 1)  # For CTC, shape of loss_fn_input: (T, N, F)For CTC, shape of loss_fn_input: (T, N, F)
    loss_fn_target = dict_targets['targets']
    feature_lengths = dict_targets['feature_lengths']
    target_lengths = dict_targets['target_lengths']
    loss = loss_fn(loss_fn_input,
                   loss_fn_target,
                   feature_lengths,
                   target_lengths)

    if reg:
        loss_reg = torch.squeeze(reg) * beta
        loss += loss_reg
    else:
        loss_reg = 0
    return loss, loss_reg


def add_meter_data(args, meter, dict_meter_data):
    outputs = dict_meter_data['net_out']
    targets = dict_meter_data['targets_metric']
    target_lengths = dict_meter_data['target_lengths']
    for i, batch in enumerate(outputs):
        outputs[i] = F.softmax(outputs[i], dim=-1).numpy()  # Shape of net_out[i]: (N, T, F)
        targets[i] = targets[i].numpy()  # Shape of net_out[i]: (N, T, F)
        target_lengths[i] = target_lengths[i].numpy()  # Shape of net_out[i]: (N, T, F)
    meter.extend_data(outputs=outputs,
                      targets=targets,
                      target_lengths=target_lengths)
    return meter


def initialize_network(net, args):
    for name, param in net.named_parameters():
        print('::: Initializing Parameters: ', name)
        if 'rnn' in name:
            if 'l0' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param[:args.hid_size, :])
                    nn.init.xavier_uniform_(param[args.hid_size:2 * args.hid_size, :])
                    nn.init.xavier_uniform_(param[2 * args.hid_size:3 * args.hid_size, :])
                    nn.init.xavier_uniform_(param[3 * args.hid_size:, :])
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param[:args.hid_size, :])
                    nn.init.orthogonal_(param[args.hid_size:2 * args.hid_size, :])
                    nn.init.orthogonal_(param[2 * args.hid_size:3 * args.hid_size, :])
                    nn.init.orthogonal_(param[3 * args.hid_size:, :])
            else:
                if 'weight' in name:
                    nn.init.orthogonal_(param[:args.hid_size, :])
                    nn.init.orthogonal_(param[args.hid_size:2 * args.hid_size, :])
                    nn.init.orthogonal_(param[2 * args.hid_size:3 * args.hid_size, :])
                    nn.init.orthogonal_(param[3 * args.hid_size:, :])
            if 'bias' in name:
                nn.init.constant_(param, 0)
        if 'fc' in name:
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

        if args.hid_type == 'LSTM':  # only LSTM biases
            if ('bias_ih' in name) or ('bias_hh' in name):
                no4 = int(len(param) / 4)
                no2 = int(len(param) / 2)
                nn.init.constant_(param, 0)
                nn.init.constant_(param[no4:no2], 1)
    print("--------------------------------------------------------------------")


def process_network(args, net, alpha):
    for name, param in net.named_parameters():
        # Quantization
        param.data = util.quantize_tensor(param.data, args.wqi, args.wqf, args.qw)
        # Column-Balanced Dropout
        if args.cbtd:
            if 'fc_extra' in name:
                if 'weight' in name:
                    util.cbtd(param.data,
                              gamma=args.gamma_fc,
                              alpha=alpha,
                              balance_pe=args.balance_pe,
                              num_pe=args.num_array_pe)
            if 'rnn' in name:
                if 'weight' in name:
                    util.cbtd(param.data,
                              gamma=args.gamma_rnn,
                              alpha=alpha,
                              balance_pe=args.balance_pe,
                              num_pe=args.num_array_pe)
    net_for_train = net
    net_for_eval = net
    return net_for_train, net_for_eval
