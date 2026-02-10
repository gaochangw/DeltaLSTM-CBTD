"""
compute Multiply-Adds(MAdd) of each leaf module
"""

import torch.nn as nn


def compute_Conv2d_madd(module, inp, out):
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    # ops per output element
    kernel_mul = k_h * k_w * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
    kernel_add_group = kernel_add * out_h * out_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    return total_mul + total_add


def compute_ConvTranspose2d_madd(module, inp, out):
    assert isinstance(module, nn.ConvTranspose2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c, in_h, in_w = inp.size()[1:]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    kernel_mul = k_h * k_w * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * in_h * in_w * (out_c // groups)
    kernel_add_group = kernel_add * in_h * in_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    return total_mul + total_add


def compute_BatchNorm2d_madd(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    in_c, in_h, in_w = inp.size()[1:]

    # 1. sub mean
    # 2. div standard deviation
    # 3. mul alpha
    # 4. add beta
    return 4 * in_c * in_h * in_w


def compute_MaxPool2d_madd(module, inp, out):
    assert isinstance(module, nn.MaxPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    if isinstance(module.kernel_size, (tuple, list)):
        k_h, k_w = module.kernel_size
    else:
        k_h, k_w = module.kernel_size, module.kernel_size
    out_c, out_h, out_w = out.size()[1:]

    return (k_h * k_w - 1) * out_h * out_w * out_c


def compute_AvgPool2d_madd(module, inp, out):
    assert isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    if isinstance(module.kernel_size, (tuple, list)):
        k_h, k_w = module.kernel_size
    else:
        k_h, k_w = module.kernel_size, module.kernel_size
    out_c, out_h, out_w = out.size()[1:]

    kernel_add = k_h * k_w - 1
    kernel_avg = 1

    return (kernel_add + kernel_avg) * (out_h * out_w) * out_c


def compute_ReLU_madd(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6))

    count = 1
    for i in inp.size()[1:]:
        count *= i
    return count


def compute_Softmax_madd(module, inp, out):
    assert isinstance(module, nn.Softmax)
    assert len(inp.size()) > 1

    count = 1
    for s in inp.size()[1:]:
        count *= s
    exp = count
    add = count - 1
    div = count
    return exp + add + div


def compute_Linear_madd(module, inp, out):
    assert isinstance(module, nn.Linear)
    # Linear layer can handle any input shape as long as last dim matches in_features
    assert inp.size(-1) == module.in_features
    assert out.size(-1) == module.out_features

    # Calculate total number of elements excluding the last dimension
    batch_elements = 1
    for dim_size in inp.size()[:-1]:
        batch_elements *= dim_size

    num_in_features = inp.size(-1)
    num_out_features = out.size(-1)

    mul = num_in_features
    add = num_in_features - 1
    return batch_elements * num_out_features * (mul + add)


def compute_Bilinear_madd(module, inp1, inp2, out):
    assert isinstance(module, nn.Bilinear)
    assert len(inp1.size()) == 2 and len(inp2.size()) == 2 and len(out.size()) == 2

    num_in_features_1 = inp1.size()[1]
    num_in_features_2 = inp2.size()[1]
    num_out_features = out.size()[1]

    mul = num_in_features_1 * num_in_features_2 + num_in_features_2
    add = num_in_features_1 * num_in_features_2 + num_in_features_2 - 1
    return num_out_features * (mul + add)


def compute_Identity_madd(module, inp, out):
    # Identity layer does no computation
    assert isinstance(module, nn.Identity)
    return 0


def compute_AdaptiveAvgPool2d_madd(module, inp, out):
    assert isinstance(module, nn.AdaptiveAvgPool2d)
    assert len(inp.size()) == 4 and len(out.size()) == 4
    
    batch_size = inp.size()[0]
    in_c, in_h, in_w = inp.size()[1:]
    out_c, out_h, out_w = out.size()[1:]
    
    # Approximate kernel size for adaptive pooling
    kernel_size = (in_h * in_w) // (out_h * out_w)
    # Add operations for averaging
    return batch_size * out_c * out_h * out_w * (kernel_size - 1 + 1)  # additions + division


def compute_Flatten_madd(module, inp, out):
    # Flatten layer does no computation, just reshaping
    assert isinstance(module, nn.Flatten)
    return 0


def compute_Sigmoid_madd(module, inp, out):
    """Compute MAdd for Sigmoid, Tanh, LogSigmoid, Softsign, Tanhshrink"""
    assert isinstance(module, (nn.Sigmoid, nn.Tanh, nn.LogSigmoid, nn.Softsign, nn.Tanhshrink))
    
    count = 1
    for i in inp.size()[1:]:
        count *= i
    
    # MAdd for these functions is generally the number of elements
    # as they involve element-wise operations
    if isinstance(module, nn.Sigmoid):
        madd = count * 2  # exp + add for denominator
    elif isinstance(module, nn.Tanh):
        madd = count * 3  # 2*exp + sub + add operations
    elif isinstance(module, nn.LogSigmoid):
        madd = count * 2  # sigmoid + log
    elif isinstance(module, nn.Softsign):
        madd = count * 2  # abs + add
    elif isinstance(module, nn.Tanhshrink):
        madd = count * 4  # tanh + sub
    else:
        madd = count * 2
    
    return madd


def compute_Complex_Activation_madd(module, inp, out):
    """Compute MAdd for GELU, SiLU/Swish, Mish, Softplus"""
    assert isinstance(module, (nn.GELU, nn.SiLU, nn.Mish, nn.Softplus))
    
    count = 1
    for i in inp.size()[1:]:
        count *= i
    
    # More complex activations involve more operations
    if isinstance(module, nn.GELU):
        madd = count * 5  # complex computation with polynomial and tanh
    elif isinstance(module, nn.SiLU):
        madd = count * 3  # sigmoid + multiply
    elif isinstance(module, nn.Mish):
        madd = count * 4  # softplus + tanh + multiply
    elif isinstance(module, nn.Softplus):
        madd = count * 2  # exp + add
    else:
        madd = count * 3
    
    return madd


def compute_Hard_Activation_madd(module, inp, out):
    """Compute MAdd for Hardsigmoid, Hardtanh, Hardswish"""
    assert isinstance(module, (nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish))
    
    count = 1
    for i in inp.size()[1:]:
        count *= i
    
    # Hard activations are computationally cheaper
    if isinstance(module, nn.Hardsigmoid):
        madd = count * 1  # add + clamp (very efficient)
    elif isinstance(module, nn.Hardtanh):
        madd = count * 1  # just clamping
    elif isinstance(module, nn.Hardswish):
        madd = count * 2  # hardsigmoid + multiply
    else:
        madd = count * 1
    
    return madd


def compute_ELU_variants_madd(module, inp, out):
    """Compute MAdd for SELU, CELU, ELU, LeakyReLU, PReLU"""
    assert isinstance(module, (nn.SELU, nn.CELU, nn.ELU, nn.LeakyReLU, nn.PReLU))
    
    count = 1
    for i in inp.size()[1:]:
        count *= i
    
    # ELU variants involve conditional operations
    if isinstance(module, (nn.SELU, nn.CELU, nn.ELU)):
        madd = count * 2  # conditional + exp operation
    elif isinstance(module, (nn.LeakyReLU, nn.PReLU)):
        madd = count * 1  # simple multiplication for negative values
    else:
        madd = count * 2
    
    return madd


def compute_madd(module, inp, out):
    if isinstance(module, nn.Conv1d):
        return compute_Conv1d_madd(module, inp, out)
    elif isinstance(module, nn.Conv2d):
        return compute_Conv2d_madd(module, inp, out)
    elif isinstance(module, nn.Conv3d):
        return compute_Conv3d_madd(module, inp, out)
    elif isinstance(module, nn.ConvTranspose1d):
        return compute_ConvTranspose1d_madd(module, inp, out)
    elif isinstance(module, nn.ConvTranspose2d):
        return compute_ConvTranspose2d_madd(module, inp, out)
    elif isinstance(module, nn.ConvTranspose3d):
        return compute_ConvTranspose3d_madd(module, inp, out)
    elif isinstance(module, nn.BatchNorm1d):
        return compute_BatchNorm1d_madd(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_madd(module, inp, out)
    elif isinstance(module, nn.BatchNorm3d):
        return compute_BatchNorm3d_madd(module, inp, out)
    elif isinstance(module, nn.MaxPool1d):
        return compute_MaxPool1d_madd(module, inp, out)
    elif isinstance(module, nn.MaxPool2d):
        return compute_MaxPool2d_madd(module, inp, out)
    elif isinstance(module, nn.MaxPool3d):
        return compute_MaxPool3d_madd(module, inp, out)
    elif isinstance(module, nn.AvgPool1d):
        return compute_AvgPool1d_madd(module, inp, out)
    elif isinstance(module, nn.AvgPool2d):
        return compute_AvgPool2d_madd(module, inp, out)
    elif isinstance(module, nn.AvgPool3d):
        return compute_AvgPool3d_madd(module, inp, out)
    elif isinstance(module, nn.AdaptiveMaxPool1d):
        return compute_AdaptiveMaxPool1d_madd(module, inp, out)
    elif isinstance(module, nn.AdaptiveMaxPool2d):
        return compute_AdaptiveMaxPool2d_madd(module, inp, out)
    elif isinstance(module, nn.AdaptiveMaxPool3d):
        return compute_AdaptiveMaxPool3d_madd(module, inp, out)
    elif isinstance(module, nn.AdaptiveAvgPool1d):
        return compute_AdaptiveAvgPool1d_madd(module, inp, out)
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return compute_AdaptiveAvgPool2d_madd(module, inp, out)
    elif isinstance(module, nn.AdaptiveAvgPool3d):
        return compute_AdaptiveAvgPool3d_madd(module, inp, out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6)):
        return compute_ReLU_madd(module, inp, out)
    elif isinstance(module, (nn.Sigmoid, nn.Tanh, nn.LogSigmoid, nn.Softsign, nn.Tanhshrink)):
        return compute_Sigmoid_madd(module, inp, out)
    elif isinstance(module, (nn.GELU, nn.SiLU, nn.Mish, nn.Softplus)):
        return compute_Complex_Activation_madd(module, inp, out)
    elif isinstance(module, (nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish)):
        return compute_Hard_Activation_madd(module, inp, out)
    elif isinstance(module, (nn.SELU, nn.CELU, nn.ELU, nn.LeakyReLU, nn.PReLU)):
        return compute_ELU_variants_madd(module, inp, out)
    elif isinstance(module, nn.Softmax):
        return compute_Softmax_madd(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_madd(module, inp, out)
    elif isinstance(module, nn.Bilinear):
        return compute_Bilinear_madd(module, inp[0], inp[1], out)
    elif isinstance(module, nn.Identity):
        return compute_Identity_madd(module, inp, out)
    elif isinstance(module, nn.Flatten):
        return compute_Flatten_madd(module, inp, out)
    elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
        return compute_Dropout_madd(module, inp, out)
    elif isinstance(module, nn.LayerNorm):
        return compute_LayerNorm_madd(module, inp, out)
    elif isinstance(module, nn.GroupNorm):
        return compute_GroupNorm_madd(module, inp, out)
    elif isinstance(module, nn.InstanceNorm1d):
        return compute_InstanceNorm1d_madd(module, inp, out)
    elif isinstance(module, nn.InstanceNorm2d):
        return compute_InstanceNorm2d_madd(module, inp, out)
    elif isinstance(module, nn.InstanceNorm3d):
        return compute_InstanceNorm3d_madd(module, inp, out)
    elif isinstance(module, nn.LocalResponseNorm):
        return compute_LocalResponseNorm_madd(module, inp, out)
    elif isinstance(module, nn.Embedding):
        return compute_Embedding_madd(module, inp, out)
    elif isinstance(module, nn.LSTM):
        return compute_LSTM_madd(module, inp, out)
    elif isinstance(module, nn.GRU):
        return compute_GRU_madd(module, inp, out)
    elif isinstance(module, nn.RNN):
        return compute_RNN_madd(module, inp, out)
    elif isinstance(module, nn.MultiheadAttention):
        return compute_MultiheadAttention_madd(module, inp, out)
    else:
        # Handle timm-specific and other custom layers by module class name
        module_name = type(module).__name__
        if module_name in ['LayerScale', 'LayerScale2d']:
            return compute_LayerScale_madd(module, inp, out)
        elif module_name == 'Flat':
            return compute_Flat_madd(module, inp, out)
        elif module_name in ['DropPath', 'StochasticDepth']:
            return compute_DropPath_madd(module, inp, out)
        elif module_name in ['FastAdaptiveAvgPool', 'FastGlobalAvgPool2d']:
            return compute_FastAdaptiveAvgPool_madd(module, inp, out)
        elif 'Scale' in module_name or 'Attention' in module_name:
            # Generic handling for scale-like or attention-like operations
            return compute_Generic_Scale_madd(module, inp, out)
        else:
            print(f"[MAdd]: {module_name} is not supported!")
            return 0


# New operator implementations
def compute_Conv1d_madd(module, inp, out):
    """Compute MAdd for Conv1d"""
    assert isinstance(module, nn.Conv1d)
    assert len(inp.size()) == 3 and len(out.size()) == 3
    
    batch_size, in_channels, input_length = inp.size()
    out_channels, out_length = out.size()[1], out.size()[2]
    kernel_size = module.kernel_size[0]
    groups = module.groups
    
    # Calculate MAdd considering groups
    input_channels_per_group = in_channels // groups
    
    # Basic convolution: output_elements * (kernel_size * input_channels_per_group + bias)
    output_elements = batch_size * out_channels * out_length
    kernel_madd = kernel_size * input_channels_per_group
    bias_madd = 1 if module.bias is not None else 0
    
    return output_elements * (kernel_madd + bias_madd)


def compute_Conv3d_madd(module, inp, out):
    """Compute MAdd for Conv3d"""
    assert isinstance(module, nn.Conv3d)
    assert len(inp.size()) == 5 and len(out.size()) == 5
    
    batch_size = inp.size()[0]
    in_channels = inp.size()[1]
    out_channels, out_d, out_h, out_w = out.size()[1], out.size()[2], out.size()[3], out.size()[4]
    
    kernel_d, kernel_h, kernel_w = module.kernel_size
    groups = module.groups
    
    # Calculate MAdd considering groups
    input_channels_per_group = in_channels // groups
    
    # Basic convolution: output_elements * (kernel_size * input_channels_per_group + bias)
    output_elements = batch_size * out_channels * out_d * out_h * out_w
    kernel_madd = kernel_d * kernel_h * kernel_w * input_channels_per_group
    bias_madd = 1 if module.bias is not None else 0
    
    return output_elements * (kernel_madd + bias_madd)


def compute_ConvTranspose1d_madd(module, inp, out):
    """Compute MAdd for ConvTranspose1d"""
    # Similar computation to Conv1d
    return compute_Conv1d_madd(module, inp, out)


def compute_ConvTranspose3d_madd(module, inp, out):
    """Compute MAdd for ConvTranspose3d"""
    # Similar computation to Conv3d
    return compute_Conv3d_madd(module, inp, out)


def compute_BatchNorm1d_madd(module, inp, out):
    """Compute MAdd for BatchNorm1d"""
    # BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta (4 ops per element)
    return inp.numel() * 4


def compute_BatchNorm3d_madd(module, inp, out):
    """Compute MAdd for BatchNorm3d"""
    return inp.numel() * 4


def compute_MaxPool1d_madd(module, inp, out):
    """Compute MAdd for MaxPool1d"""
    kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
    return out.numel() * kernel_size


def compute_MaxPool3d_madd(module, inp, out):
    """Compute MAdd for MaxPool3d"""
    if isinstance(module.kernel_size, int):
        kernel_size = module.kernel_size ** 3
    else:
        kernel_size = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
    return out.numel() * kernel_size


def compute_AvgPool1d_madd(module, inp, out):
    """Compute MAdd for AvgPool1d"""
    kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
    return out.numel() * (kernel_size + 1)  # sum + division


def compute_AvgPool3d_madd(module, inp, out):
    """Compute MAdd for AvgPool3d"""
    if isinstance(module.kernel_size, int):
        kernel_size = module.kernel_size ** 3
    else:
        kernel_size = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
    return out.numel() * (kernel_size + 1)  # sum + division


def compute_AdaptiveMaxPool1d_madd(module, inp, out):
    """Compute MAdd for AdaptiveMaxPool1d"""
    input_length = inp.size(-1)
    output_length = out.size(-1)
    avg_kernel_size = max(input_length // output_length, 1)
    return out.numel() * avg_kernel_size


def compute_AdaptiveMaxPool2d_madd(module, inp, out):
    """Compute MAdd for AdaptiveMaxPool2d"""
    input_h, input_w = inp.size()[-2:]
    output_h, output_w = out.size()[-2:]
    avg_kernel_size = max((input_h * input_w) // (output_h * output_w), 1)
    return out.numel() * avg_kernel_size


def compute_AdaptiveMaxPool3d_madd(module, inp, out):
    """Compute MAdd for AdaptiveMaxPool3d"""
    input_d, input_h, input_w = inp.size()[-3:]
    output_d, output_h, output_w = out.size()[-3:]
    avg_kernel_size = max((input_d * input_h * input_w) // (output_d * output_h * output_w), 1)
    return out.numel() * avg_kernel_size


def compute_AdaptiveAvgPool1d_madd(module, inp, out):
    """Compute MAdd for AdaptiveAvgPool1d"""
    input_length = inp.size(-1)
    output_length = out.size(-1)
    avg_kernel_size = max(input_length // output_length, 1)
    return out.numel() * (avg_kernel_size + 1)  # sum + division


def compute_AdaptiveAvgPool3d_madd(module, inp, out):
    """Compute MAdd for AdaptiveAvgPool3d"""
    input_d, input_h, input_w = inp.size()[-3:]
    output_d, output_h, output_w = out.size()[-3:]
    avg_kernel_size = max((input_d * input_h * input_w) // (output_d * output_h * output_w), 1)
    return out.numel() * (avg_kernel_size + 1)  # sum + division


def compute_Dropout_madd(module, inp, out):
    """Compute MAdd for Dropout layers"""
    # Dropout during inference has no computational cost
    return 0


def compute_LayerNorm_madd(module, inp, out):
    """Compute MAdd for LayerNorm"""
    # LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta (4 ops per element)
    return inp.numel() * 4


def compute_GroupNorm_madd(module, inp, out):
    """Compute MAdd for GroupNorm"""
    return inp.numel() * 4


def compute_InstanceNorm1d_madd(module, inp, out):
    """Compute MAdd for InstanceNorm1d"""
    return inp.numel() * 4


def compute_InstanceNorm2d_madd(module, inp, out):
    """Compute MAdd for InstanceNorm2d"""
    return inp.numel() * 4


def compute_InstanceNorm3d_madd(module, inp, out):
    """Compute MAdd for InstanceNorm3d"""
    return inp.numel() * 4


def compute_LocalResponseNorm_madd(module, inp, out):
    """Compute MAdd for LocalResponseNorm"""
    return inp.numel() * 6  # More complex normalization


def compute_Embedding_madd(module, inp, out):
    """Compute MAdd for Embedding"""
    # Embedding is essentially a lookup table
    return out.numel()


def compute_LSTM_madd(module, inp, out):
    """Compute MAdd for LSTM (simplified)"""
    if len(inp.size()) == 3:
        batch_size, seq_len, input_size = inp.size()
    else:
        batch_size, input_size = inp.size()
        seq_len = 1
    
    hidden_size = module.hidden_size
    num_layers = module.num_layers
    
    # LSTM: 4 gates * (input_size + hidden_size + 1) operations per cell
    ops_per_cell = 4 * (input_size + hidden_size + 1) * hidden_size
    total_ops = batch_size * seq_len * num_layers * ops_per_cell
    
    return total_ops


def compute_GRU_madd(module, inp, out):
    """Compute MAdd for GRU (simplified)"""
    if len(inp.size()) == 3:
        batch_size, seq_len, input_size = inp.size()
    else:
        batch_size, input_size = inp.size()
        seq_len = 1
    
    hidden_size = module.hidden_size
    num_layers = module.num_layers
    
    # GRU: 3 gates * (input_size + hidden_size + 1) operations per cell
    ops_per_cell = 3 * (input_size + hidden_size + 1) * hidden_size
    total_ops = batch_size * seq_len * num_layers * ops_per_cell
    
    return total_ops


def compute_RNN_madd(module, inp, out):
    """Compute MAdd for vanilla RNN (simplified)"""
    if len(inp.size()) == 3:
        batch_size, seq_len, input_size = inp.size()
    else:
        batch_size, input_size = inp.size()
        seq_len = 1
    
    hidden_size = module.hidden_size
    num_layers = module.num_layers
    
    # Simple RNN: (input_size + hidden_size + 1) operations per cell
    ops_per_cell = (input_size + hidden_size + 1) * hidden_size
    total_ops = batch_size * seq_len * num_layers * ops_per_cell
    
    return total_ops


def compute_MultiheadAttention_madd(module, inp, out):
    """Compute MAdd for MultiheadAttention (simplified)"""
    if isinstance(inp, (list, tuple)):
        query = inp[0]
    else:
        query = inp
        
    if len(query.size()) == 3:
        batch_size, seq_len, embed_dim = query.size()
    else:
        batch_size, embed_dim = query.size()
        seq_len = 1
    
    num_heads = module.num_heads
    
    # Simplified: Q, K, V projections + attention + output projection
    projection_ops = 3 * seq_len * embed_dim * embed_dim
    attention_ops = num_heads * seq_len * seq_len * (embed_dim // num_heads)
    output_ops = seq_len * embed_dim * embed_dim
    
    total_ops = batch_size * (projection_ops + attention_ops + output_ops)
    return total_ops


# TIMM-specific and custom layer implementations
def compute_LayerScale_madd(module, inp, out):
    """Compute MAdd for LayerScale/LayerScale2d"""
    # LayerScale applies element-wise scaling: x * gamma
    # This is a simple element-wise multiplication
    return inp.numel()


def compute_Flat_madd(module, inp, out):
    """Compute MAdd for Flat operation"""
    # Flat is similar to Flatten, just reshaping with no computation
    return 0


def compute_DropPath_madd(module, inp, out):
    """Compute MAdd for DropPath/StochasticDepth"""
    # DropPath during inference has no computational cost
    # During training it involves random dropout of entire samples
    return 0


def compute_FastAdaptiveAvgPool_madd(module, inp, out):
    """Compute MAdd for FastAdaptiveAvgPool/FastGlobalAvgPool2d from TIMM"""
    # FastAdaptiveAvgPool is typically a global average pooling operation
    # It averages over spatial dimensions H×W to produce a 1×1 output
    input_size = inp.numel()
    output_size = out.numel()
    
    # For global average pooling: sum all spatial elements then divide
    # Number of elements being averaged = input_size / output_size
    if output_size > 0:
        avg_elements = input_size // output_size
        return avg_elements * output_size  # additions for averaging
    else:
        return 0


def compute_Generic_Scale_madd(module, inp, out):
    """Generic handler for scale-like operations"""
    # Most scale operations are element-wise multiplications
    if hasattr(module, 'weight') or hasattr(module, 'scale'):
        return inp.numel()  # Element-wise scaling
    else:
        return 0  # No computation
