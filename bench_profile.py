"""Profile to find actual bottleneck"""
import time, torch
from nnmodels.deltalstm import DeltaRNN
from nnmodels.deltalstm_optimized import DeltaRNNOptimized

torch.manual_seed(42)
device = 'cuda:0'
batch, seq, hid, inp = 64, 98, 256, 123

kwargs = {'aqi': 8, 'aqf': 8, 'nqi': 2, 'nqf': 4, 'th_x': 0.2, 'th_h': 0.2,
          'num_array': 4, 'num_array_pe': 64, 'hardsigmoid': 0, 'hardtanh': 0}

x = torch.randn(seq, batch, inp, device=device)
fl = torch.full((batch,), seq, dtype=torch.long, device=device)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Config: batch={batch}, seq={seq}, hidden={hid}\n")

# Test just forward pass (no backward)
print("=== Forward pass only (no backward) ===")
m1 = DeltaRNN(mode='DeltaLSTM', input_size=inp, hidden_size=hid, num_layers=1, **kwargs).to(device)
m2 = DeltaRNNOptimized(mode='DeltaLSTM', input_size=inp, hidden_size=hid, num_layers=1, **kwargs).to(device)
m2.load_state_dict(m1.state_dict())

m1.train()
m2.train()

# Warmup
for _ in range(3):
    with torch.no_grad():
        m1(x, quantize=1, feature_lengths=fl)
        m2(x, quantize=1, feature_lengths=fl)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    with torch.no_grad():
        m1(x, quantize=1, feature_lengths=fl)
torch.cuda.synchronize()
t1 = (time.perf_counter() - t0) / 20

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    with torch.no_grad():
        m2(x, quantize=1, feature_lengths=fl)
torch.cuda.synchronize()
t2 = (time.perf_counter() - t0) / 20

print(f"Original:  {t1*1000:.1f}ms")
print(f"Optimized: {t2*1000:.1f}ms | {t1/t2:.2f}x speedup")

# Profile with PyTorch profiler
print("\n=== PyTorch Profiler (Original) ===")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        m1(x, quantize=1, feature_lengths=fl)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("\n=== PyTorch Profiler (Optimized) ===")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        m2(x, quantize=1, feature_lengths=fl)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
