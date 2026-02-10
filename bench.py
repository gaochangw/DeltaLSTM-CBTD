"""Minimal benchmark - no torch.compile"""
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

def bench(model, n=10, warmup=3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(warmup):
        opt.zero_grad()
        out, st, reg = model(x, quantize=1, feature_lengths=fl)
        (out.sum() + reg.sum()).backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        opt.zero_grad()
        out, st, reg = model(x, quantize=1, feature_lengths=fl)
        (out.sum() + reg.sum()).backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Config: batch={batch}, seq={seq}, hidden={hid}")

m1 = DeltaRNN(mode='DeltaLSTM', input_size=inp, hidden_size=hid, num_layers=1, **kwargs).to(device)
t1 = bench(m1)
print(f"Original:  {t1*1000:.1f}ms | {batch/t1:.0f} samp/s")

m2 = DeltaRNNOptimized(mode='DeltaLSTM', input_size=inp, hidden_size=hid, num_layers=1, **kwargs).to(device)
m2.load_state_dict(m1.state_dict())
t2 = bench(m2)
print(f"Optimized: {t2*1000:.1f}ms | {batch/t2:.0f} samp/s | {t1/t2:.2f}x")
