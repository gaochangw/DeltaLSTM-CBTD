# TorchStat2: PyTorch Model Analyzer

TorchStat2 is a comprehensive and lightweight neural network analyzer for PyTorch models. It provides detailed statistics about your neural networks, including computational complexity, memory usage, and performance analysis. This enhanced version extends the original torchstat with additional features for modern deep learning workflows.

## What TorchStat2 Analyzes

- **Parameters**: Total number of trainable and non-trainable parameters
- **FLOPs**: Theoretical floating-point operations per forward pass
- **MADDs**: Multiply-accumulate operations (more accurate for some architectures)
- **Memory Usage**: Inference memory requirements and memory access patterns
- **Execution Time**: Layer-wise execution duration and performance bottlenecks
- **Throughput**: Model throughput analysis across different batch sizes
- **Memory Access**: Detailed memory read/write patterns for optimization

## Installation

```bash
git clone https://github.com/Phoenix8215/torchstat2.git
cd torchstat2
python setup.py install
```

## Quick Start

### Python API Usage

#### Basic Model Analysis
```python
import torch
import torchvision.models as models
from torchstat2 import stat

# Load a pre-trained model
model = models.resnet18(pretrained=True)

# Analyze the model
stat(model, (3, 224, 224))
```

#### Advanced Analysis with Throughput
```python
from torchstat2 import stat_with_throughput
import torchvision.models as models

model = models.resnet50()

# Combined analysis: model statistics + throughput benchmarking
results = stat_with_throughput(
    model=model,
    input_size=(3, 224, 224),
    device='cuda',  # or 'cpu'
    batch_sizes=[1, 8, 16, 32, 64]
)
```

#### Throughput-Only Analysis
```python
from torchstat2 import throughput, compare_models_throughput
import torchvision.models as models

# Single model throughput analysis
model = models.efficientnet_b0()
results = throughput(
    model=model,
    input_shape=(3, 224, 224),
    device='cuda',
    batch_sizes=[1, 4, 8, 16, 32],
    warmup_time=2.0,
    test_time=10.0
)

# Compare multiple models
models_dict = {
    'ResNet18': models.resnet18(),
    'ResNet50': models.resnet50(),
    'EfficientNet-B0': models.efficientnet_b0()
}

comparison = compare_models_throughput(
    models_dict=models_dict,
    input_shape=(3, 224, 224),
    device='cuda',
    batch_sizes=[1, 8, 16, 32]
)
```

#### Batch Size Scaling Analysis
```python
from torchstat2 import analyze_batch_scaling

model = models.resnet34()
scaling_results = analyze_batch_scaling(
    model=model,
    input_shape=(3, 224, 224),
    device='cuda',
    max_batch_size=128,
    model_name="ResNet34"
)
```

### Command Line Interface

#### Basic Usage
```bash
# Analyze a model defined in a Python file
torchstat2 -f model_definition.py -m ModelClassName

# Specify custom input size
torchstat2 -f model_definition.py -m ModelClassName -s 3x512x512

# Example with a custom model file
torchstat2 -f examples/custom_model.py -m MyCustomNet -s 3x224x224
```

#### Example model file (custom_model.py)
```python
import torch.nn as nn

class MyCustomNet(nn.Module):
    def __init__(self):
        super(MyCustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1000)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## Example Output

### Model Statistics Output
```
                                    module name input shape output shape     params memory(MB)        MAdd duration[%]       Flops  MemRead(B)  MemWrite(B)    MemR+W(B)
0                                         conv1   3 224 224   64 224 224      1,792       12.25  115,605,504      2.86%  57,802,752      614400     12845056    13459456
1                                          bn1  64 224 224   64 224 224        128       12.25            0      0.89%           0      102400     12845056    12947456
2                                         relu  64 224 224   64 224 224          0       12.25            0      1.24%           0           0     12845056    12845056
3                                        pool   64 224 224       64 1 1          0        0.00      3211264      0.45%     3211264    12845056          256    12845312
4                                          fc      64     1000      65,000        0.25       127,000      0.89%      64,000       260256         4000       264256
total                                                                        66,920       37.00  118,943,768    100.00%  61,077,016    13822112     38534424    52356536
========================================================================================================================================
Total params: 66,920
----------------------------------------------------------------------------------------------------------------------------------------
Total memory: 37.00MB
Total MAdd: 118.94MMAdd
Total Flops: 61.08MFlops
Total MemR+W: 49.94MB
```

### Throughput Analysis Output
```
============================================================
Throughput Analysis: ResNet18
Device: cuda
Input shape: (3, 224, 224)
============================================================
Batch size   1: 245.67 images/s (avg: 4.07ms, std: 0.12ms)
Batch size   8: 1456.23 images/s (avg: 5.49ms, std: 0.08ms)
Batch size  16: 2134.45 images/s (avg: 7.50ms, std: 0.15ms)
Batch size  32: 2456.78 images/s (avg: 13.02ms, std: 0.23ms)

Optimal batch size: 32 (max throughput: 2456.78 images/s)
```

## Advanced Features

### Custom Query Granularity
```python
from torchstat2 import ModelStat

# Create analyzer with custom granularity
analyzer = ModelStat(model, (3, 224, 224), query_granularity=2)
analyzer.show_report()
```

### Direct Statistics Access
```python
from torchstat2 import ModelStat

analyzer = ModelStat(model, (3, 224, 224))
collected_nodes = analyzer._analyze_model()

# Access individual layer statistics
for node in collected_nodes:
    print(f"Layer: {node.name}")
    print(f"Parameters: {node.parameter_quantity}")
    print(f"FLOPs: {node.Flops}")
    print(f"Memory: {node.inference_memory} MB")
```

### Supported Layers

TorchStat2 supports analysis of 40+ PyTorch layer types including:

**Convolution Layers**: Conv1d, Conv2d, Conv3d, ConvTranspose1d/2d/3d
**Normalization**: BatchNorm1d/2d/3d, LayerNorm, GroupNorm, InstanceNorm1d/2d/3d
**Activation Functions**: ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, and many more
**Pooling**: AvgPool, MaxPool, AdaptiveAvgPool, AdaptiveMaxPool (1d/2d/3d)
**Linear Layers**: Linear, Identity, Flatten
**Regularization**: Dropout, Dropout1d/2d/3d
**Utility**: Upsample, and more

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is an enhanced version based on the original [torchstat](https://github.com/Swall0w/torchstat) repository. We extend our gratitude to the original authors for providing the foundation for this improved analysis tool.

---

**Happy Analyzing! 🎉**

*TorchStat2 - Making PyTorch model analysis comprehensive, accurate, and insightful.*