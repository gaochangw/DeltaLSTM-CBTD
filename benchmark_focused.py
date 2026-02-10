"""
Focused benchmark to isolate the impact of specific optimizations.
"""
import time
import torch
import torch.nn as nn
import numpy as np
from nnmodels.deltalstm import DeltaRNN
from nnmodels.deltalstm_optimized import DeltaRNNOptimized


def benchmark_training_only(model, x, feature_lengths, num_iterations=100, warmup=10):
    """Benchmark training pass only (the main bottleneck)."""
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
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output, state, reg = model(x, quantize=1, feature_lengths=feature_lengths)
        loss = output.sum() + reg.sum()
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

    # GSCDv2-like config (the actual use case)
    batch_size = 64
    seq_len = 98
    hidden_size = 256
    input_size = 123
    num_layers = 1

    kwargs = {
        'aqi': 8, 'aqf': 8,
        'nqi': 2, 'nqf': 4,
        'th_x': 0.2, 'th_h': 0.2,
        'num_array': 4, 'num_array_pe': 64,
        'hardsigmoid': 0, 'hardtanh': 0
    }

    # Create test data
    x = torch.randn(seq_len, batch_size, input_size, device=device)
    feature_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

    print(f"\nConfig: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")
    print("="*60)

    # Test original
    print("\n1. Original DeltaLSTM:")
    model_orig = DeltaRNN(
        mode='DeltaLSTM',
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        **kwargs
    ).to(device)

    time_orig = benchmark_training_only(model_orig, x, feature_lengths, num_iterations=50)
    samples_per_sec_orig = batch_size / time_orig
    print(f"   Time: {time_orig*1000:.2f} ms/iter")
    print(f"   Throughput: {samples_per_sec_orig:.1f} samples/sec")

    # Test optimized
    print("\n2. Optimized DeltaLSTM:")
    model_opt = DeltaRNNOptimized(
        mode='DeltaLSTM',
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        **kwargs
    ).to(device)
    model_opt.load_state_dict(model_orig.state_dict())

    time_opt = benchmark_training_only(model_opt, x, feature_lengths, num_iterations=50)
    samples_per_sec_opt = batch_size / time_opt
    print(f"   Time: {time_opt*1000:.2f} ms/iter")
    print(f"   Throughput: {samples_per_sec_opt:.1f} samples/sec")
    print(f"   Speedup: {time_orig/time_opt:.2f}x")

    # Skip torch.compile for now - it takes too long to compile
    # # Test with torch.compile (PyTorch 2.x)
    # print("\n3. Optimized DeltaLSTM + torch.compile:")
    # try:
    #     model_compiled = torch.compile(model_opt, mode="reduce-overhead")
    #     time_compiled = benchmark_training_only(model_compiled, x, feature_lengths, num_iterations=50, warmup=20)
    #     samples_per_sec_compiled = batch_size / time_compiled
    #     print(f"   Time: {time_compiled*1000:.2f} ms/iter")
    #     print(f"   Throughput: {samples_per_sec_compiled:.1f} samples/sec")
    #     print(f"   Speedup vs original: {time_orig/time_compiled:.2f}x")
    #     print(f"   Speedup vs optimized: {time_opt/time_compiled:.2f}x")
    # except Exception as e:
    #     print(f"   torch.compile failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original:    {samples_per_sec_orig:.1f} samples/sec")
    print(f"Optimized:   {samples_per_sec_opt:.1f} samples/sec ({time_orig/time_opt:.2f}x speedup)")

    # Now test actual GSCDv2 training loop timing components
    print("\n" + "="*60)
    print("COMPONENT ANALYSIS")
    print("="*60)

    # Test forward-only (no backward)
    model_orig.train()
    model_opt.train()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            output, state, reg = model_orig(x, quantize=1, feature_lengths=feature_lengths)
    torch.cuda.synchronize()
    time_forward_orig = (time.perf_counter() - start) / 100

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            output, state, reg = model_opt(x, quantize=1, feature_lengths=feature_lengths)
    torch.cuda.synchronize()
    time_forward_opt = (time.perf_counter() - start) / 100

    print(f"Forward pass only:")
    print(f"  Original:  {time_forward_orig*1000:.2f} ms")
    print(f"  Optimized: {time_forward_opt*1000:.2f} ms")
    print(f"  Speedup:   {time_forward_orig/time_forward_opt:.2f}x")


if __name__ == '__main__':
    main()
