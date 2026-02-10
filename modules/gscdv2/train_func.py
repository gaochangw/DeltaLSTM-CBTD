"""
Training functions for Google Speech Commands Dataset v2 - Keyword Spotting.
Uses cross-entropy loss for classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.util as util


def get_batch_data(args, dict_batch_array):
    """Prepare batch data for training/evaluation."""
    features = dict_batch_array['features']  # (T, B, F)
    feature_lengths = dict_batch_array['feature_lengths']
    labels = dict_batch_array['labels']  # (B,)

    # Move to device (CUDA, MPS, or CPU)
    device = getattr(args, 'device', torch.device('cpu'))
    features = features.to(device)
    labels = labels.to(device)
    feature_lengths = feature_lengths.to(device)

    dict_batch_tensor = {
        'features': features,
        'feature_lengths': feature_lengths,
        'labels': labels,
        'targets': labels,
        'targets_metric': labels,
        'target_lengths': torch.ones(len(labels), dtype=torch.int32)
    }

    return dict_batch_tensor


def forward_propagation(net, dict_batch_tensor, state=None):
    """Forward propagation through the network."""
    outputs, state, reg = net(dict_batch_tensor['features'], state, dict_batch_tensor['feature_lengths'])
    out_final, out_rnn = outputs

    # For KWS classification, use mean pooling over time dimension
    # out_final shape: (B, T, n_classes) or (T, B, n_classes) depending on model
    if out_final.dim() == 3:
        # Average over time dimension
        if out_final.shape[0] == dict_batch_tensor['features'].shape[0]:
            # Shape is (T, B, n_classes)
            out_final = out_final.mean(dim=0)  # (B, n_classes)
        else:
            # Shape is (B, T, n_classes)
            out_final = out_final.mean(dim=1)  # (B, n_classes)

    return out_final, state, reg


def calculate_loss(loss_fn, net_out, dict_targets, reg, beta):
    """Calculate cross-entropy loss for classification."""
    labels = dict_targets['labels']

    # Cross-entropy loss
    loss = loss_fn(net_out, labels)

    # Add regularization if present
    if reg is not None and beta > 0:
        loss_reg = torch.squeeze(reg) * beta
        loss = loss + loss_reg
    else:
        loss_reg = 0

    return loss, loss_reg


def add_meter_data(args, meter, dict_meter_data):
    """Add batch data to meter for evaluation."""
    outputs = dict_meter_data['net_out']  # List of batch outputs
    labels = dict_meter_data['targets_metric']  # List of batch labels

    for i in range(len(outputs)):
        # Get predictions
        if outputs[i].dim() == 3:
            # Average over time if not already done
            batch_out = outputs[i].mean(dim=1 if outputs[i].shape[1] > outputs[i].shape[0] else 0)
        else:
            batch_out = outputs[i]

        probs = F.softmax(batch_out, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        # Pass probabilities to meter for AUC-ROC calculation
        meter.add_batch(
            preds.cpu().numpy(),
            labels[i].cpu().numpy(),
            probs.detach().cpu().numpy()
        )

    return meter


def initialize_network(net, args):
    """Initialize network parameters."""
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
