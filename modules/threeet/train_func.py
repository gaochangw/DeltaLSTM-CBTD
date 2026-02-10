"""
Training functions for 3ET+ Event-Based Eye Tracking - Regression task.
Uses weighted MSE loss for pupil center (x, y) prediction.
"""
import torch
import torch.nn as nn
import utils.util as util


def get_batch_data(args, dict_batch_array):
    """Prepare batch data for training/evaluation."""
    features = dict_batch_array['features']           # (T, B, C, H, W)
    feature_lengths = dict_batch_array['feature_lengths']
    targets = dict_batch_array['targets']              # (B, T, 2)
    targets_full = dict_batch_array['targets_full']    # (B, T, 3)

    # Move to device
    device = getattr(args, 'device', torch.device('cpu'))
    features = features.to(device)
    targets = targets.to(device)
    targets_full = targets_full.to(device)
    feature_lengths = feature_lengths.to(device)

    dict_batch_tensor = {
        'features': features,
        'feature_lengths': feature_lengths,
        'targets': targets,               # (B, T, 2) for loss
        'targets_full': targets_full,      # (B, T, 3) for metrics
        'targets_metric': targets_full,    # (B, T, 3) for metric computation
        'labels': targets,                 # Alias for compatibility
        'target_lengths': torch.full((features.shape[1],), features.shape[0],
                                     dtype=torch.int32)
    }

    return dict_batch_tensor


def forward_propagation(net, dict_batch_tensor, state=None):
    """Forward propagation through the network."""
    outputs, state, reg = net(dict_batch_tensor['features'], state,
                              dict_batch_tensor['feature_lengths'])
    out_final, out_rnn = outputs

    # out_final shape: (B, T, 2) — per-frame xy predictions
    # No mean pooling (unlike GSCDv2 classification)

    return out_final, state, reg


def calculate_loss(loss_fn, net_out, dict_targets, reg, beta):
    """Calculate weighted MSE loss for regression."""
    targets = dict_targets['targets']  # (B, T, 2)

    # Compute loss
    loss = loss_fn(net_out, targets)

    # Add regularization if present
    if reg is not None and beta > 0:
        loss_reg = torch.squeeze(reg) * beta
        loss = loss + loss_reg
    else:
        loss_reg = 0

    return loss, loss_reg


def add_meter_data(args, meter, dict_meter_data):
    """Add batch data to meter for evaluation."""
    outputs = dict_meter_data['net_out']          # List of (B, T, 2) tensors
    targets = dict_meter_data['targets_metric']   # List of (B, T, 3) tensors

    for i in range(len(outputs)):
        meter.add_batch(
            outputs[i].cpu(),     # (B, T, 2) predictions
            targets[i].cpu()      # (B, T, 3) targets with close flag
        )

    return meter


def initialize_network(net, args):
    """Initialize network parameters."""
    if args.hid_type == 'JaneEyeNet':
        # JaneEyeNet-specific initialization
        for name, param in net.named_parameters():
            print('::: Initializing Parameters: ', name)
            if 'conv' in name and 'weight' in name and param.dim() >= 2:
                # Kaiming normal for Conv layers in JaneEyeNet blocks
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'conv' in name and 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'linear_head' in name:
                # Xavier for detection head
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                if 'bias' in name:
                    nn.init.constant_(param, 0)
            # ConvLSTM and MLP use default PyTorch initialization
        print("--------------------------------------------------------------------")
        return

    for name, param in net.named_parameters():
        print('::: Initializing Parameters: ', name)
        if 'cnn' in name:
            # Kaiming initialization for CNN layers
            if 'weight' in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            if 'bias' in name:
                nn.init.constant_(param, 0)
        elif 'rnn' in name:
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

        # LSTM forget gate bias initialization
        if args.hid_type == 'LSTM':
            if ('bias_ih' in name) or ('bias_hh' in name):
                no4 = int(len(param) / 4)
                no2 = int(len(param) / 2)
                nn.init.constant_(param, 0)
                nn.init.constant_(param[no4:no2], 1)

    print("--------------------------------------------------------------------")


def process_network(args, net, alpha):
    """Process network for quantization and CBTD."""
    # JaneEyeNet and RVT skip quantization and CBTD
    if args.hid_type in ('JaneEyeNet', 'RVT'):
        return net, net

    for name, param in net.named_parameters():
        # Quantization
        param.data = util.quantize_tensor(param.data, args.wqi, args.wqf, args.qw)

        # Column-Balanced Targeted Dropout
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
