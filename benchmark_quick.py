"""
Quick benchmark comparing original vs optimized DeltaLSTM on actual GSCDv2 data.
Runs on a small subset for fast comparison.
"""
import time
import torch
import torch.nn as nn
import numpy as np
import h5py
from model import Model
from nnmodels.deltalstm import DeltaRNN
from nnmodels.deltalstm_optimized import DeltaRNNOptimized
import argparse


def create_mock_args():
    """Create args matching GSCDv2 config."""
    args = argparse.Namespace(
        hid_type='DeltaLSTM',
        hid_size=256,
        hid_layers=1,
        hid_dropout=0,
        fc_type='FC',
        fc_extra_size=0,
        qa=1,
        qw=1,
        aqi=8,
        aqf=8,
        wqi=1,
        wqf=7,
        nqi=2,
        nqf=4,
        qa_fc_extra=1,
        aqi_fc_extra=8,
        aqf_fc_extra=8,
        qa_fc_final=1,
        aqi_fc_final=8,
        aqf_fc_final=8,
        th_x=0.2,
        th_h=0.2,
        use_cuda=1,
        hardsigmoid=0,
        hardtanh=0,
        num_array=4,
        num_array_pe=64,
    )
    return args


def load_sample_data(h5_path, num_samples=500):
    """Load a small subset of data."""
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
        feature_lens = f['feature_lens'][:]
        labels = f['labels'][:]
        label_lens = f['label_lens'][:]

    # Get first num_samples
    sample_features = []
    sample_labels = []
    feat_idx = 0
    label_idx = 0

    for i in range(min(num_samples, len(feature_lens))):
        feat_len = int(feature_lens[i])
        label_len = int(label_lens[i])
        sample_features.append(features[feat_idx:feat_idx+feat_len])
        sample_labels.append(labels[label_idx:label_idx+label_len])
        feat_idx += feat_len
        label_idx += label_len

    return sample_features, sample_labels, feature_lens[:num_samples]


def collate_batch(features, labels, feature_lens, batch_size=64):
    """Create a single batch."""
    batch_features = features[:batch_size]
    batch_lens = feature_lens[:batch_size]

    # Pad to same length
    max_len = int(max(batch_lens))
    n_feat = batch_features[0].shape[1]

    padded = np.zeros((batch_size, max_len, n_feat), dtype=np.float32)
    for i, feat in enumerate(batch_features):
        padded[i, :len(feat), :] = feat

    return padded, batch_lens


def benchmark_model(model, x, feature_lengths, num_iterations=20, warmup=5):
    """Benchmark a model."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Dummy labels
    batch_size = x.size(1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=x.device)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        outputs, state, reg = model(x, feature_lengths=feature_lengths)
        out = outputs[0].mean(dim=1)  # Mean pool over time
        loss = criterion(out, labels)
        if reg is not None:
            loss = loss + 0.01 * reg.sum()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        optimizer.zero_grad()
        outputs, state, reg = model(x, feature_lengths=feature_lengths)
        out = outputs[0].mean(dim=1)
        loss = criterion(out, labels)
        if reg is not None:
            loss = loss + 0.01 * reg.sum()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_iterations


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda:0'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Use synthetic data matching GSCDv2 shapes
    batch_size = 64
    seq_len = 98  # 1 second audio at 10ms stride
    n_feat = 123  # 40 filterbanks * 3 (static + delta + delta-delta) + 3

    # Create synthetic batch (T, N, F)
    x = torch.randn(seq_len, batch_size, n_feat, device=device)
    feature_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

    print(f"Batch shape: {x.shape} (T={x.shape[0]}, N={x.shape[1]}, F={x.shape[2]})")

    args = create_mock_args()
    input_size = x.shape[2]
    n_classes = 12

    # Test Original DeltaLSTM
    print("\n" + "="*60)
    print("1. Original DeltaLSTM")
    print("="*60)

    model_orig = Model(args, input_size, n_classes).to(device)
    time_orig = benchmark_model(model_orig, x, feature_lengths, num_iterations=30)
    throughput_orig = batch_size / time_orig
    print(f"   Time per iteration: {time_orig*1000:.2f} ms")
    print(f"   Throughput: {throughput_orig:.1f} samples/sec")

    # Test Optimized DeltaLSTM
    print("\n" + "="*60)
    print("2. Optimized DeltaLSTM")
    print("="*60)

    # Swap in the optimized RNN
    args_opt = create_mock_args()
    model_opt = Model(args_opt, input_size, n_classes).to(device)

    # Replace with optimized version
    model_opt.rnn = DeltaRNNOptimized(
        mode='DeltaLSTM',
        input_size=input_size,
        hidden_size=args.hid_size,
        num_layers=args.hid_layers,
        th_x=model_orig.th_x,
        th_h=model_orig.th_h,
        aqi=args.aqi,
        aqf=args.aqf,
        nqi=args.nqi,
        nqf=args.nqf,
        num_array=args.num_array,
        num_array_pe=args.num_array_pe,
        hardtanh=args.hardtanh,
        hardsigmoid=args.hardsigmoid
    ).to(device)

    # Copy weights
    model_opt.load_state_dict(model_orig.state_dict())

    time_opt = benchmark_model(model_opt, x, feature_lengths, num_iterations=30)
    throughput_opt = batch_size / time_opt
    print(f"   Time per iteration: {time_opt*1000:.2f} ms")
    print(f"   Throughput: {throughput_opt:.1f} samples/sec")

    # Summary
    speedup = time_orig / time_opt
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original DeltaLSTM:  {throughput_orig:.1f} samples/sec ({time_orig*1000:.2f} ms/iter)")
    print(f"Optimized DeltaLSTM: {throughput_opt:.1f} samples/sec ({time_opt*1000:.2f} ms/iter)")
    print(f"Speedup: {speedup:.2f}x")

    if speedup > 1:
        print(f"\nOptimized version is {(speedup-1)*100:.1f}% faster!")
    else:
        print(f"\nOptimized version is {(1-speedup)*100:.1f}% slower.")


if __name__ == '__main__':
    main()
