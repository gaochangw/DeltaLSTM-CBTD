"""
Benchmark script to compare original vs optimized DeltaLSTM implementations.

Tests:
1. Training throughput (samples/sec)
2. Numerical accuracy (output difference)
3. Memory usage
"""
import time
import torch
import torch.nn as nn
import numpy as np
from nnmodels.deltalstm import DeltaRNN
from nnmodels.deltalstm_optimized import DeltaRNNOptimized


def create_test_data(batch_size, seq_len, input_size, device):
    """Create random test data."""
    x = torch.randn(seq_len, batch_size, input_size, device=device)
    # Feature lengths needed for stats computation
    feature_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    return x, feature_lengths


def benchmark_forward_backward(model, x, feature_lengths, num_iterations=100, warmup=10):
    """Benchmark forward + backward pass."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        output, state, reg = model(x, quantize=1, feature_lengths=feature_lengths)
        loss = output.sum() + reg.sum()
        loss.backward()
        optimizer.step()

    # Sync before timing
    if x.is_cuda:
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output, state, reg = model(x, quantize=1, feature_lengths=feature_lengths)
        loss = output.sum() + reg.sum()
        loss.backward()
        optimizer.step()

    if x.is_cuda:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / num_iterations


def benchmark_forward_only(model, x, feature_lengths, num_iterations=100, warmup=10):
    """Benchmark forward pass only (inference)."""
    model.eval()
    model.reset_stats()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model.reset_stats()
            output, state, reg = model(x, quantize=1, feature_lengths=feature_lengths)

    # Sync before timing
    if x.is_cuda:
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            model.reset_stats()
            output, state, reg = model(x, quantize=1, feature_lengths=feature_lengths)

    if x.is_cuda:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / num_iterations


def check_numerical_accuracy(model_orig, model_opt, x, feature_lengths):
    """Check if both models produce the same output."""
    model_orig.eval()
    model_opt.eval()
    model_orig.reset_stats()
    model_opt.reset_stats()

    with torch.no_grad():
        out_orig, state_orig, reg_orig = model_orig(x, quantize=1, feature_lengths=feature_lengths)
        model_opt.reset_stats()
        out_opt, state_opt, reg_opt = model_opt(x, quantize=1, feature_lengths=feature_lengths)

    # Compare outputs
    max_diff = (out_orig - out_opt).abs().max().item()
    mean_diff = (out_orig - out_opt).abs().mean().item()
    # Handle reg potentially being on different devices (original bug)
    reg_diff = abs(reg_orig.cpu().item() - reg_opt.cpu().item())

    return max_diff, mean_diff, reg_diff


def run_benchmark(batch_size, seq_len, hidden_size, input_size, num_layers, device, num_iterations=50):
    """Run full benchmark comparison."""
    print(f"\n{'='*70}")
    print(f"Configuration: batch={batch_size}, seq={seq_len}, hidden={hidden_size}, "
          f"input={input_size}, layers={num_layers}, device={device}")
    print(f"{'='*70}")

    # Common kwargs for DeltaRNN
    kwargs = {
        'aqi': 8, 'aqf': 8,
        'nqi': 2, 'nqf': 4,
        'th_x': 0.2, 'th_h': 0.2,
        'num_array': 4, 'num_array_pe': 64,
        'hardsigmoid': 0, 'hardtanh': 0
    }

    # Create models
    model_orig = DeltaRNN(
        mode='DeltaLSTM',
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        **kwargs
    ).to(device)

    model_opt = DeltaRNNOptimized(
        mode='DeltaLSTM',
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        **kwargs
    ).to(device)

    # Copy weights from original to optimized
    model_opt.load_state_dict(model_orig.state_dict())

    # Create test data
    x, feature_lengths = create_test_data(batch_size, seq_len, input_size, device)

    # Check numerical accuracy first
    print("\n1. Numerical Accuracy Check:")
    max_diff, mean_diff, reg_diff = check_numerical_accuracy(model_orig, model_opt, x, feature_lengths)
    print(f"   Output max diff:  {max_diff:.2e}")
    print(f"   Output mean diff: {mean_diff:.2e}")
    print(f"   Reg diff:         {reg_diff:.2e}")

    # Benchmark training (forward + backward)
    print(f"\n2. Training Benchmark ({num_iterations} iterations):")

    time_orig_train = benchmark_forward_backward(model_orig, x, feature_lengths, num_iterations)
    samples_per_sec_orig = batch_size / time_orig_train
    print(f"   Original:  {time_orig_train*1000:.2f} ms/iter, {samples_per_sec_orig:.1f} samples/sec")

    time_opt_train = benchmark_forward_backward(model_opt, x, feature_lengths, num_iterations)
    samples_per_sec_opt = batch_size / time_opt_train
    print(f"   Optimized: {time_opt_train*1000:.2f} ms/iter, {samples_per_sec_opt:.1f} samples/sec")

    speedup_train = time_orig_train / time_opt_train
    print(f"   Speedup:   {speedup_train:.2f}x")

    # Benchmark inference (forward only)
    print(f"\n3. Inference Benchmark ({num_iterations} iterations):")

    time_orig_infer = benchmark_forward_only(model_orig, x, feature_lengths, num_iterations)
    samples_per_sec_orig_infer = batch_size / time_orig_infer
    print(f"   Original:  {time_orig_infer*1000:.2f} ms/iter, {samples_per_sec_orig_infer:.1f} samples/sec")

    time_opt_infer = benchmark_forward_only(model_opt, x, feature_lengths, num_iterations)
    samples_per_sec_opt_infer = batch_size / time_opt_infer
    print(f"   Optimized: {time_opt_infer*1000:.2f} ms/iter, {samples_per_sec_opt_infer:.1f} samples/sec")

    speedup_infer = time_orig_infer / time_opt_infer
    print(f"   Speedup:   {speedup_infer:.2f}x")

    return {
        'config': f"b{batch_size}_s{seq_len}_h{hidden_size}",
        'speedup_train': speedup_train,
        'speedup_infer': speedup_infer,
        'samples_orig_train': samples_per_sec_orig,
        'samples_opt_train': samples_per_sec_opt,
        'samples_orig_infer': samples_per_sec_orig_infer,
        'samples_opt_infer': samples_per_sec_opt_infer,
        'max_diff': max_diff,
    }


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Detect device
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    print("\n" + "="*70)
    print("DeltaLSTM Optimization Benchmark")
    print("="*70)

    results = []

    # Test different configurations
    configs = [
        # GSCDv2-like config
        {'batch_size': 64, 'seq_len': 98, 'hidden_size': 256, 'input_size': 123, 'num_layers': 1},
        # Larger hidden size
        {'batch_size': 64, 'seq_len': 98, 'hidden_size': 512, 'input_size': 123, 'num_layers': 1},
        # TIMIT-like config (larger)
        {'batch_size': 32, 'seq_len': 200, 'hidden_size': 1024, 'input_size': 123, 'num_layers': 2},
        # Longer sequence
        {'batch_size': 32, 'seq_len': 500, 'hidden_size': 256, 'input_size': 123, 'num_layers': 1},
    ]

    for config in configs:
        result = run_benchmark(
            device=device,
            num_iterations=50,
            **config
        )
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<25} {'Train Speedup':>15} {'Infer Speedup':>15} {'Max Diff':>12}")
    print("-"*70)
    for r in results:
        print(f"{r['config']:<25} {r['speedup_train']:>14.2f}x {r['speedup_infer']:>14.2f}x {r['max_diff']:>12.2e}")

    avg_train_speedup = np.mean([r['speedup_train'] for r in results])
    avg_infer_speedup = np.mean([r['speedup_infer'] for r in results])
    print("-"*70)
    print(f"{'Average':<25} {avg_train_speedup:>14.2f}x {avg_infer_speedup:>14.2f}x")


if __name__ == '__main__':
    main()
