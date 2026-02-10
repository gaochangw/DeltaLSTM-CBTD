import torch.nn as nn
import torch
import numpy as np


def compute_flops(module, inp, out):
    if isinstance(module, nn.Conv1d):
        return compute_Conv1d_flops(module, inp, out)
    elif isinstance(module, nn.Conv2d):
        return compute_Conv2d_flops(module, inp, out)
    elif isinstance(module, nn.Conv3d):
        return compute_Conv3d_flops(module, inp, out)
    elif isinstance(module, nn.ConvTranspose1d):
        return compute_ConvTranspose1d_flops(module, inp, out)
    elif isinstance(module, nn.ConvTranspose2d):
        return compute_ConvTranspose2d_flops(module, inp, out)
    elif isinstance(module, nn.ConvTranspose3d):
        return compute_ConvTranspose3d_flops(module, inp, out)
    elif isinstance(module, nn.BatchNorm1d):
        return compute_BatchNorm1d_flops(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_flops(module, inp, out)
    elif isinstance(module, nn.BatchNorm3d):
        return compute_BatchNorm3d_flops(module, inp, out)
    elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        return compute_Pool_flops(module, inp, out)
    elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)):
        return compute_AdaptivePool_flops(module, inp, out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU)):
        return compute_ReLU_flops(module, inp, out)
    elif isinstance(module, (nn.Sigmoid, nn.Tanh, nn.LogSigmoid, nn.Softsign, nn.Tanhshrink)):
        return compute_Sigmoid_flops(module, inp, out)
    elif isinstance(module, (nn.GELU, nn.SiLU, nn.Mish, nn.Softplus)):
        return compute_Complex_Activation_flops(module, inp, out)
    elif isinstance(module, (nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish)):
        return compute_Hard_Activation_flops(module, inp, out)
    elif isinstance(module, (nn.SELU, nn.CELU)):
        return compute_ELU_variants_flops(module, inp, out)
    elif isinstance(module, nn.Softmax):
        return compute_Softmax_flops(module, inp, out)
    elif isinstance(module, nn.Upsample):
        return compute_Upsample_flops(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_flops(module, inp, out)
    elif isinstance(module, nn.Identity):
        return compute_Identity_flops(module, inp, out)
    elif isinstance(module, nn.Flatten):
        return compute_Flatten_flops(module, inp, out)
    elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
        return compute_Dropout_flops(module, inp, out)
    elif isinstance(module, nn.LayerNorm):
        return compute_LayerNorm_flops(module, inp, out)
    elif isinstance(module, nn.GroupNorm):
        return compute_GroupNorm_flops(module, inp, out)
    elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        return compute_InstanceNorm_flops(module, inp, out)
    elif isinstance(module, nn.LocalResponseNorm):
        return compute_LocalResponseNorm_flops(module, inp, out)
    elif isinstance(module, nn.Embedding):
        return compute_Embedding_flops(module, inp, out)
    elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        return compute_RNN_flops(module, inp, out)
    elif isinstance(module, nn.MultiheadAttention):
        return compute_MultiheadAttention_flops(module, inp, out)
    else:
        # Handle timm-specific and other custom layers by module class name
        module_name = type(module).__name__
        if module_name in ['LayerScale', 'LayerScale2d']:
            return compute_LayerScale_flops(module, inp, out)
        elif module_name == 'Flat':
            return compute_Flat_flops(module, inp, out)
        elif module_name in ['DropPath', 'StochasticDepth']:
            return compute_DropPath_flops(module, inp, out)
        elif module_name in ['FastAdaptiveAvgPool', 'FastGlobalAvgPool2d']:
            return compute_FastAdaptiveAvgPool_flops(module, inp, out)
        elif 'Scale' in module_name or 'Attention' in module_name:
            # Generic handling for scale-like or attention-like operations
            return compute_Generic_Scale_flops(module, inp, out)
        else:
            print(f"[Flops]: {module_name} is not supported!")
            return 0
    pass


def compute_Conv2d_flops(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    # Calculate FLOPs considering groups - using MAC operations for consistency
    input_channels_per_group = in_c // groups
    
    # FLOPs = output_elements * (kernel_size * input_channels_per_group * 2 - 1 + bias)
    output_elements = batch_size * out_c * out_h * out_w
    kernel_flops = k_h * k_w * input_channels_per_group * 2 - 1  # MAC operations per output
    bias_flops = 1 if module.bias is not None else 0
    
    return output_elements * (kernel_flops + bias_flops)


def compute_BatchNorm2d_flops(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    
    # BatchNorm operations per element:
    # 1. Subtract mean: x - mean (1 op)
    # 2. Divide by std: (x - mean) / sqrt(var + eps) (1 op, sqrt is precomputed)
    # 3. If affine: multiply by weight (1 op) and add bias (1 op)
    
    num_elements = inp.numel()
    base_ops = 2  # subtract mean + divide by std
    
    if module.affine:
        affine_ops = 2  # multiply by weight + add bias
    else:
        affine_ops = 0
    
    return num_elements * (base_ops + affine_ops)


def compute_ReLU_flops(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU))
    # Activation functions work element-wise on any tensor shape
    active_elements_count = inp.numel()
    return active_elements_count


def compute_Linear_flops(module, inp, out):
    assert isinstance(module, nn.Linear)
    # Linear layer can handle any input shape as long as last dim matches in_features
    # Input: (..., in_features), Output: (..., out_features)
    assert inp.size(-1) == module.in_features
    assert out.size(-1) == module.out_features
    
    # Calculate total number of elements excluding the last dimension
    batch_elements = 1
    for dim_size in inp.size()[:-1]:
        batch_elements *= dim_size
    
    # FLOPs = batch_elements * (in_features * out_features * 2 - out_features + bias)
    # Using MAC operations for consistency with convolutions
    output_elements = batch_elements * out.size(-1)
    kernel_flops = inp.size(-1) * 2 - 1  # MAC operations per output
    bias_flops = 1 if module.bias is not None else 0
    
    return output_elements * (kernel_flops + bias_flops)

def compute_Upsample_flops(module, inp, out):
    assert isinstance(module, nn.Upsample)
    # For upsample, consider the actual output tensor shape including channels
    if isinstance(out, (tuple, list)):
        output_tensor = out[0]
    else:
        output_tensor = out
    
    # The computation cost depends on the upsampling mode
    # For nearest neighbor: minimal computation (just indexing)
    # For linear/bilinear/trilinear: interpolation requires computation
    
    if module.mode == 'nearest':
        # Nearest neighbor requires minimal computation - just indexing
        return 0
    elif module.mode in ['linear', 'bilinear', 'trilinear', 'bicubic']:
        # Interpolation methods require computation proportional to output elements
        # Each output element requires interpolation calculations
        output_elements = output_tensor.numel()
        if module.mode == 'linear':
            # Linear interpolation: 2 points, 1 multiplication per output
            return output_elements
        elif module.mode == 'bilinear':
            # Bilinear interpolation: 4 points, ~3 multiplications per output  
            return output_elements * 3
        elif module.mode == 'trilinear':
            # Trilinear interpolation: 8 points, ~7 multiplications per output
            return output_elements * 7
        elif module.mode == 'bicubic':
            # Bicubic interpolation: 16 points, ~15 multiplications per output
            return output_elements * 15
    else:
        # Default case for other modes
        return output_tensor.numel()


def compute_Identity_flops(module, inp, out):
    # Identity layer does no computation
    assert isinstance(module, nn.Identity)
    return 0


def compute_Flatten_flops(module, inp, out):
    # Flatten layer does no computation, just reshaping
    assert isinstance(module, nn.Flatten)
    return 0


# New operators for FLOPs calculation
def compute_Conv1d_flops(module, inp, out):
    """Compute FLOPs for Conv1d"""
    assert isinstance(module, nn.Conv1d)
    assert len(inp.size()) == 3 and len(out.size()) == 3
    
    batch_size, in_channels, input_length = inp.size()
    out_channels, out_length = out.size()[1], out.size()[2]
    kernel_size = module.kernel_size[0]
    groups = module.groups
    
    # Calculate FLOPs considering groups
    input_channels_per_group = in_channels // groups
    
    # FLOPs = output_elements * (kernel_size * input_channels_per_group * 2 - 1 + bias)
    output_elements = batch_size * out_channels * out_length
    kernel_flops = kernel_size * input_channels_per_group * 2 - 1  # MAC operations per output
    bias_flops = 1 if module.bias is not None else 0
    
    return output_elements * (kernel_flops + bias_flops)


def compute_Conv3d_flops(module, inp, out):
    """Compute FLOPs for Conv3d"""
    assert isinstance(module, nn.Conv3d)
    assert len(inp.size()) == 5 and len(out.size()) == 5
    
    batch_size = inp.size()[0]
    in_channels = inp.size()[1]
    out_channels, out_d, out_h, out_w = out.size()[1], out.size()[2], out.size()[3], out.size()[4]
    
    kernel_d, kernel_h, kernel_w = module.kernel_size
    groups = module.groups
    
    # Calculate FLOPs considering groups
    input_channels_per_group = in_channels // groups
    
    # FLOPs = output_elements * (kernel_size * input_channels_per_group * 2 - 1 + bias)
    output_elements = batch_size * out_channels * out_d * out_h * out_w
    kernel_flops = kernel_d * kernel_h * kernel_w * input_channels_per_group * 2 - 1
    bias_flops = 1 if module.bias is not None else 0
    
    return output_elements * (kernel_flops + bias_flops)


def compute_ConvTranspose1d_flops(module, inp, out):
    """Compute FLOPs for ConvTranspose1d"""
    return compute_Conv1d_flops(module, inp, out)


def compute_ConvTranspose2d_flops(module, inp, out):
    """Compute FLOPs for ConvTranspose2d"""
    return compute_Conv2d_flops(module, inp, out)


def compute_ConvTranspose3d_flops(module, inp, out):
    """Compute FLOPs for ConvTranspose3d"""
    return compute_Conv3d_flops(module, inp, out)


def compute_BatchNorm1d_flops(module, inp, out):
    """Compute FLOPs for BatchNorm1d"""
    assert isinstance(module, nn.BatchNorm1d)
    
    # BatchNorm operations per element:
    # 1. Subtract mean: x - mean (1 op)
    # 2. Divide by std: (x - mean) / sqrt(var + eps) (1 op, sqrt is precomputed)
    # 3. If affine: multiply by weight (1 op) and add bias (1 op)
    
    num_elements = inp.numel()
    base_ops = 2  # subtract mean + divide by std
    
    if module.affine:
        affine_ops = 2  # multiply by weight + add bias
    else:
        affine_ops = 0
    
    return num_elements * (base_ops + affine_ops)


def compute_BatchNorm3d_flops(module, inp, out):
    """Compute FLOPs for BatchNorm3d"""
    assert isinstance(module, nn.BatchNorm3d)
    
    # BatchNorm operations per element:
    # 1. Subtract mean: x - mean (1 op)
    # 2. Divide by std: (x - mean) / sqrt(var + eps) (1 op, sqrt is precomputed)
    # 3. If affine: multiply by weight (1 op) and add bias (1 op)
    
    num_elements = inp.numel()
    base_ops = 2  # subtract mean + divide by std
    
    if module.affine:
        affine_ops = 2  # multiply by weight + add bias
    else:
        affine_ops = 0
    
    return num_elements * (base_ops + affine_ops)


def compute_Pool_flops(module, inp, out):
    """Compute FLOPs for pooling operations"""
    if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        if hasattr(module, 'kernel_size'):
            if isinstance(module.kernel_size, int):
                if isinstance(module, nn.MaxPool1d):
                    kernel_size = module.kernel_size
                elif isinstance(module, nn.MaxPool2d):
                    kernel_size = module.kernel_size ** 2
                else:  # MaxPool3d
                    kernel_size = module.kernel_size ** 3
            else:
                kernel_size = 1
                for k in module.kernel_size:
                    kernel_size *= k
        else:
            kernel_size = 1
        return out.numel() * (kernel_size - 1)
    
    elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        if hasattr(module, 'kernel_size'):
            if isinstance(module.kernel_size, int):
                if isinstance(module, nn.AvgPool1d):
                    kernel_size = module.kernel_size
                elif isinstance(module, nn.AvgPool2d):
                    kernel_size = module.kernel_size ** 2
                else:  # AvgPool3d
                    kernel_size = module.kernel_size ** 3
            else:
                kernel_size = 1
                for k in module.kernel_size:
                    kernel_size *= k
        else:
            kernel_size = 1
        # Average pooling: (kernel_size - 1) additions + 1 division = kernel_size operations
        return out.numel() * kernel_size


def compute_AdaptivePool_flops(module, inp, out):
    """Compute FLOPs for adaptive pooling"""
    if isinstance(module, (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)):
        if isinstance(module, nn.AdaptiveMaxPool1d):
            input_size = inp.size(-1)
            output_size = out.size(-1)
            # For each output position, we need to find max among corresponding input positions
            # Use ceiling division to handle non-divisible cases
            kernel_size = (input_size + output_size - 1) // output_size
            # Max pooling: kernel_size - 1 comparisons per output element
            return out.numel() * (kernel_size - 1)
        elif isinstance(module, nn.AdaptiveMaxPool2d):
            input_h, input_w = inp.size()[-2:]
            output_h, output_w = out.size()[-2:]
            # Calculate average kernel size for each dimension
            kernel_h = (input_h + output_h - 1) // output_h
            kernel_w = (input_w + output_w - 1) // output_w
            kernel_size = kernel_h * kernel_w
            return out.numel() * (kernel_size - 1)
        else:  # AdaptiveMaxPool3d
            input_d, input_h, input_w = inp.size()[-3:]
            output_d, output_h, output_w = out.size()[-3:]
            kernel_d = (input_d + output_d - 1) // output_d
            kernel_h = (input_h + output_h - 1) // output_h
            kernel_w = (input_w + output_w - 1) // output_w
            kernel_size = kernel_d * kernel_h * kernel_w
            return out.numel() * (kernel_size - 1)
    
    elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
        if isinstance(module, nn.AdaptiveAvgPool1d):
            input_size = inp.size(-1)
            output_size = out.size(-1)
            # Use ceiling division to handle non-divisible cases
            kernel_size = (input_size + output_size - 1) // output_size
            # Average pooling: (kernel_size - 1) additions + 1 division = kernel_size operations
            return out.numel() * kernel_size
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            input_h, input_w = inp.size()[-2:]
            output_h, output_w = out.size()[-2:]
            kernel_h = (input_h + output_h - 1) // output_h
            kernel_w = (input_w + output_w - 1) // output_w
            kernel_size = kernel_h * kernel_w
            return out.numel() * kernel_size
        else:  # AdaptiveAvgPool3d
            input_d, input_h, input_w = inp.size()[-3:]
            output_d, output_h, output_w = out.size()[-3:]
            kernel_d = (input_d + output_d - 1) // output_d
            kernel_h = (input_h + output_h - 1) // output_h
            kernel_w = (input_w + output_w - 1) // output_w
            kernel_size = kernel_d * kernel_h * kernel_w
            return out.numel() * kernel_size


def compute_Dropout_flops(module, inp, out):
    """Compute FLOPs for Dropout"""
    return 0


def compute_LayerNorm_flops(module, inp, out):
    """Compute FLOPs for LayerNorm"""
    # LayerNorm operations:
    # 1. Compute mean: sum(elements)/n = n-1 additions + 1 division
    # 2. Compute variance: sum((x-mean)^2)/n = n subtractions + n multiplications + n-1 additions + 1 division
    # 3. Normalize: (x-mean)/sqrt(var+eps) = n subtractions + n divisions (sqrt is precomputed)
    # 4. If affine: n multiplications + n additions
    
    # Get normalized shape elements
    normalized_elements = 1
    for dim in module.normalized_shape:
        normalized_elements *= dim
    
    # Number of normalization groups (batch * other dimensions)
    num_groups = inp.numel() // normalized_elements
    
    # Per group operations
    mean_ops = normalized_elements  # n-1 additions + 1 division ≈ n
    var_ops = normalized_elements * 3  # n sub + n mul + n-1 add + 1 div ≈ 3n
    normalize_ops = normalized_elements * 2  # n sub + n div
    
    base_flops = num_groups * (mean_ops + var_ops + normalize_ops)
    
    # Affine transformation if enabled
    if module.elementwise_affine:
        affine_flops = inp.numel() * 2  # multiply by weight + add bias
    else:
        affine_flops = 0
    
    return base_flops + affine_flops


def compute_GroupNorm_flops(module, inp, out):
    """Compute FLOPs for GroupNorm"""
    # GroupNorm operations are similar to LayerNorm but across groups
    batch_size = inp.size()[0]
    num_channels = inp.size()[1]
    num_groups = module.num_groups
    channels_per_group = num_channels // num_groups
    
    # Calculate spatial elements (H*W for 2D, H*W*D for 3D, etc.)
    spatial_elements = inp.numel() // (batch_size * num_channels)
    elements_per_group = channels_per_group * spatial_elements
    
    # Operations per group normalization
    mean_ops = elements_per_group  # n-1 additions + 1 division ≈ n
    var_ops = elements_per_group * 3  # n sub + n mul + n-1 add + 1 div ≈ 3n
    normalize_ops = elements_per_group * 2  # n sub + n div
    
    total_groups = batch_size * num_groups
    base_flops = total_groups * (mean_ops + var_ops + normalize_ops)
    
    # Affine transformation if enabled
    if module.affine:
        affine_flops = inp.numel() * 2  # multiply by weight + add bias
    else:
        affine_flops = 0
    
    return base_flops + affine_flops


def compute_InstanceNorm_flops(module, inp, out):
    """Compute FLOPs for InstanceNorm"""
    # InstanceNorm normalizes each channel of each sample independently
    batch_size = inp.size()[0]
    num_channels = inp.size()[1]
    
    # Calculate spatial elements (H*W for 2D, H*W*D for 3D, etc.)
    spatial_elements = inp.numel() // (batch_size * num_channels)
    
    # Operations per instance (one channel of one sample)
    mean_ops = spatial_elements  # n-1 additions + 1 division ≈ n
    var_ops = spatial_elements * 3  # n sub + n mul + n-1 add + 1 div ≈ 3n
    normalize_ops = spatial_elements * 2  # n sub + n div
    
    num_instances = batch_size * num_channels
    base_flops = num_instances * (mean_ops + var_ops + normalize_ops)
    
    # Affine transformation if enabled
    if module.affine:
        affine_flops = inp.numel() * 2  # multiply by weight + add bias
    else:
        affine_flops = 0
    
    return base_flops + affine_flops


def compute_LocalResponseNorm_flops(module, inp, out):
    """Compute FLOPs for LocalResponseNorm"""
    return inp.numel() * 8


def compute_Embedding_flops(module, inp, out):
    """Compute FLOPs for Embedding"""
    # Embedding is a lookup table operation - pure memory access with no arithmetic operations
    # No FLOPs should be counted for table lookups
    return 0


def compute_RNN_flops(module, inp, out):
    """Compute FLOPs for RNN layers"""
    if len(inp.size()) == 3:
        batch_size, seq_len, input_size = inp.size()
    else:
        batch_size, input_size = inp.size()
        seq_len = 1
    
    hidden_size = module.hidden_size
    num_layers = module.num_layers
    
    if isinstance(module, nn.LSTM):
        ops_per_cell = 4 * (input_size + hidden_size + 1) * hidden_size * 2
    elif isinstance(module, nn.GRU):
        ops_per_cell = 3 * (input_size + hidden_size + 1) * hidden_size * 2
    else:  # vanilla RNN
        ops_per_cell = (input_size + hidden_size + 1) * hidden_size * 2
    
    return batch_size * seq_len * num_layers * ops_per_cell


def compute_MultiheadAttention_flops(module, inp, out):
    """Compute FLOPs for MultiheadAttention with proper kdim/vdim support"""
    # Handle different input formats: single tensor, tuple of (Q,K,V), or list
    if isinstance(inp, (list, tuple)):
        if len(inp) >= 3:
            # Separate Q, K, V inputs
            query, key, value = inp[0], inp[1], inp[2]
        elif len(inp) == 2:
            # Q separate, K=V (cross-attention)
            query, key = inp[0], inp[1]
            value = key
        else:
            # Single input (self-attention)
            query = key = value = inp[0]
    else:
        # Single tensor input (self-attention)
        query = key = value = inp
    
    # Get dimensions from tensors
    if len(query.size()) == 3:
        batch_size, q_seq_len, q_embed_dim = query.size()
    else:
        batch_size, q_embed_dim = query.size()
        q_seq_len = 1
    
    if len(key.size()) == 3:
        k_seq_len, k_embed_dim = key.size(1), key.size(2)
    else:
        k_seq_len, k_embed_dim = 1, key.size(-1)
        
    if len(value.size()) == 3:
        v_seq_len, v_embed_dim = value.size(1), value.size(2)
    else:
        v_seq_len, v_embed_dim = 1, value.size(-1)
    
    # Get module dimensions, considering kdim/vdim parameters
    embed_dim = module.embed_dim
    num_heads = module.num_heads
    head_dim = embed_dim // num_heads
    
    # Use kdim/vdim if specified, otherwise use embed_dim
    kdim = getattr(module, 'kdim', None) or k_embed_dim
    vdim = getattr(module, 'vdim', None) or v_embed_dim
    
    # 1. Linear projections for Q, K, V (including bias if present)
    # Q projection: q_seq_len * q_embed_dim -> q_seq_len * embed_dim
    q_proj_flops = q_seq_len * q_embed_dim * (embed_dim * 2 - 1)
    if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
        q_proj_flops += q_seq_len * embed_dim  # bias addition
    elif hasattr(module, 'q_proj_weight') and hasattr(module, 'q_proj_bias') and module.q_proj_bias is not None:
        q_proj_flops += q_seq_len * embed_dim
    
    # K projection: k_seq_len * kdim -> k_seq_len * embed_dim
    k_proj_flops = k_seq_len * kdim * (embed_dim * 2 - 1)
    if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
        k_proj_flops += k_seq_len * embed_dim
    elif hasattr(module, 'k_proj_weight') and hasattr(module, 'k_proj_bias') and module.k_proj_bias is not None:
        k_proj_flops += k_seq_len * embed_dim
        
    # V projection: v_seq_len * vdim -> v_seq_len * embed_dim  
    v_proj_flops = v_seq_len * vdim * (embed_dim * 2 - 1)
    if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
        v_proj_flops += v_seq_len * embed_dim
    elif hasattr(module, 'v_proj_weight') and hasattr(module, 'v_proj_bias') and module.v_proj_bias is not None:
        v_proj_flops += v_seq_len * embed_dim
    
    # 2. Attention computation: Q @ K^T
    # Shape: [batch, num_heads, q_seq_len, head_dim] @ [batch, num_heads, head_dim, k_seq_len]
    # = [batch, num_heads, q_seq_len, k_seq_len]
    qk_flops = num_heads * q_seq_len * k_seq_len * (head_dim * 2 - 1)
    
    # 3. Attention weights @ V
    # Shape: [batch, num_heads, q_seq_len, k_seq_len] @ [batch, num_heads, k_seq_len, head_dim]
    # = [batch, num_heads, q_seq_len, head_dim]
    attention_v_flops = num_heads * q_seq_len * head_dim * (k_seq_len * 2 - 1)
    
    # 4. Output projection: [q_seq_len, embed_dim] -> [q_seq_len, embed_dim]
    output_proj_flops = q_seq_len * embed_dim * (embed_dim * 2 - 1)
    if hasattr(module, 'out_proj') and hasattr(module.out_proj, 'bias') and module.out_proj.bias is not None:
        output_proj_flops += q_seq_len * embed_dim
    
    # 5. Softmax computation (approximate)
    # For each attention head: q_seq_len * k_seq_len elements
    # Softmax: exp + sum + divide ≈ 3 ops per element
    softmax_flops = num_heads * q_seq_len * k_seq_len * 3
    
    total_flops = (q_proj_flops + k_proj_flops + v_proj_flops + 
                   qk_flops + attention_v_flops + output_proj_flops + softmax_flops)
    
    return batch_size * total_flops


# TIMM-specific and custom layer implementations for Flops
def compute_LayerScale_flops(module, inp, out):
    """Compute FLOPs for LayerScale/LayerScale2d"""
    # LayerScale: element-wise multiplication
    return inp.numel()


def compute_Flat_flops(module, inp, out):
    """Compute FLOPs for Flat operation"""
    # Flat is just reshaping, no computation
    return 0


def compute_DropPath_flops(module, inp, out):
    """Compute FLOPs for DropPath/StochasticDepth"""
    # No computational cost during inference
    return 0


def compute_FastAdaptiveAvgPool_flops(module, inp, out):
    """Compute FLOPs for FastAdaptiveAvgPool/FastGlobalAvgPool2d from TIMM"""
    # FastAdaptiveAvgPool is typically a global average pooling operation
    # It averages over spatial dimensions H×W to produce a 1×1 output
    input_size = inp.numel()
    output_size = out.numel()
    
    # For global average pooling: sum all spatial elements then divide
    # Number of elements being averaged = input_size / output_size
    if output_size > 0:
        avg_elements = input_size // output_size
        # Additions for summing + divisions for averaging
        return (avg_elements - 1) * output_size + output_size
    else:
        return 0


def compute_Generic_Scale_flops(module, inp, out):
    """Generic handler for scale-like operations (FLOPs)"""
    if hasattr(module, 'weight') or hasattr(module, 'scale'):
        return inp.numel()  # Element-wise operation
    else:
        return 0


def compute_Sigmoid_flops(module, inp, out):
    """Compute FLOPs for Sigmoid, Tanh, LogSigmoid, Softsign, Tanhshrink"""
    assert isinstance(module, (nn.Sigmoid, nn.Tanh, nn.LogSigmoid, nn.Softsign, nn.Tanhshrink))
    # Activation functions work element-wise on any tensor shape
    active_elements_count = inp.numel()
    
    # These activation functions typically involve exponential operations
    # Sigmoid: 1/(1+exp(-x)) - approximately 4 operations per element
    # Tanh: (exp(x)-exp(-x))/(exp(x)+exp(-x)) - approximately 6 operations
    # LogSigmoid: log(sigmoid(x)) - approximately 5 operations
    # Softsign: x/(1+|x|) - approximately 3 operations
    # Tanhshrink: x - tanh(x) - approximately 7 operations
    
    if isinstance(module, nn.Sigmoid):
        ops_per_element = 4  # exp, add, div
    elif isinstance(module, nn.Tanh):
        ops_per_element = 6  # 2*exp, sub, add, div
    elif isinstance(module, nn.LogSigmoid):
        ops_per_element = 5  # exp, add, div, log
    elif isinstance(module, nn.Softsign):
        ops_per_element = 3  # abs, add, div
    elif isinstance(module, nn.Tanhshrink):
        ops_per_element = 7  # tanh(6) + sub(1)
    else:
        ops_per_element = 4  # default
    
    return active_elements_count * ops_per_element


def compute_Complex_Activation_flops(module, inp, out):
    """Compute FLOPs for GELU, SiLU/Swish, Mish, Softplus"""
    assert isinstance(module, (nn.GELU, nn.SiLU, nn.Mish, nn.Softplus))
    # Activation functions work element-wise on any tensor shape
    active_elements_count = inp.numel()
    
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) - approximately 10 operations
    # SiLU/Swish: x * sigmoid(x) - approximately 5 operations  
    # Mish: x * tanh(softplus(x)) - approximately 8 operations
    # Softplus: log(1 + exp(x)) - approximately 3 operations
    
    if isinstance(module, nn.GELU):
        ops_per_element = 10  # complex computation with tanh and polynomial
    elif isinstance(module, nn.SiLU):
        ops_per_element = 5   # sigmoid + multiply
    elif isinstance(module, nn.Mish):
        ops_per_element = 8   # softplus + tanh + multiply
    elif isinstance(module, nn.Softplus):
        ops_per_element = 3   # exp, add, log
    else:
        ops_per_element = 5   # default
    
    return active_elements_count * ops_per_element


def compute_Hard_Activation_flops(module, inp, out):
    """Compute FLOPs for Hardsigmoid, Hardtanh, Hardswish"""
    assert isinstance(module, (nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish))
    # Activation functions work element-wise on any tensor shape
    active_elements_count = inp.numel()
    
    # Hard activations use piecewise linear functions, much cheaper than smooth versions
    # Hardsigmoid: max(0, min(1, (x + 3) / 6)) - approximately 2 operations
    # Hardtanh: max(-1, min(1, x)) - approximately 1 operation (clamping)
    # Hardswish: x * hardsigmoid(x) - approximately 3 operations
    
    if isinstance(module, nn.Hardsigmoid):
        ops_per_element = 2   # add, div, clamp
    elif isinstance(module, nn.Hardtanh):
        ops_per_element = 1   # clamp
    elif isinstance(module, nn.Hardswish):
        ops_per_element = 3   # hardsigmoid + multiply
    else:
        ops_per_element = 2   # default
    
    return active_elements_count * ops_per_element


def compute_ELU_variants_flops(module, inp, out):
    """Compute FLOPs for SELU, CELU variants of ELU"""
    assert isinstance(module, (nn.SELU, nn.CELU))
    # Activation functions work element-wise on any tensor shape
    active_elements_count = inp.numel()
    
    # SELU: scale * (max(0, x) + min(0, alpha * (exp(x/alpha) - 1)))
    # CELU: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    # Both involve conditional exp operations, similar to ELU but with scaling
    
    ops_per_element = 4  # conditional, exp, sub, mul/scale
    return active_elements_count * ops_per_element


def compute_Softmax_flops(module, inp, out):
    """Compute FLOPs for Softmax"""
    assert isinstance(module, nn.Softmax)
    assert len(inp.size()) > 1
    
    # Softmax computation often uses the numerically stable form:
    #   exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    # This involves computing a max, subtracting it from each element,
    # then performing exponentiation, summation, and division.
    
    # Get the softmax dimension
    softmax_dim = module.dim
    if softmax_dim < 0:
        softmax_dim = len(inp.size()) + softmax_dim
    
    # Number of elements in the softmax dimension
    softmax_size = inp.size(softmax_dim)
    
    # Total number of elements
    total_elements = inp.numel()
    
    # Operations per softmax computation:
    # - exp for each element: softmax_size operations
    # - sum across dimension: (softmax_size - 1) additions  
    # - division for each element: softmax_size operations
    # - max across dimension: (softmax_size - 1) comparisons
    # - subtraction for each element: softmax_size operations
    # Total: ops_per_softmax = exp + sum + div + max + sub
    ops_per_softmax = (
        softmax_size                        # exp
        + (softmax_size - 1)               # sum
        + softmax_size                     # div
        + (softmax_size - 1)               # max comparisons
        + softmax_size                     # subtraction
    )
    
    # Number of independent softmax computations
    num_softmax_groups = total_elements // softmax_size
    
    return num_softmax_groups * ops_per_softmax
