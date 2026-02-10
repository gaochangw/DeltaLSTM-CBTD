import torch.nn as nn
import torch


def compute_memory(module, inp, out):
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU)):
        return compute_ReLU_memory(module, inp, out)
    elif isinstance(module, nn.PReLU):
        return compute_PReLU_memory(module, inp, out)
    elif isinstance(module, (nn.Sigmoid, nn.Tanh, nn.LogSigmoid, nn.Softsign, nn.Tanhshrink)):
        return compute_Activation_memory(module, inp, out)
    elif isinstance(module, (nn.GELU, nn.SiLU, nn.Mish, nn.Softplus)):
        return compute_Activation_memory(module, inp, out)
    elif isinstance(module, (nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish)):
        return compute_Activation_memory(module, inp, out)
    elif isinstance(module, (nn.SELU, nn.CELU)):
        return compute_ELU_memory(module, inp, out)
    elif isinstance(module, nn.Softmax):
        return compute_Softmax_memory(module, inp, out)
    elif isinstance(module, nn.Conv1d):
        return compute_Conv1d_memory(module, inp, out)
    elif isinstance(module, nn.Conv2d):
        return compute_Conv2d_memory(module, inp, out)
    elif isinstance(module, nn.Conv3d):
        return compute_Conv3d_memory(module, inp, out)
    elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return compute_ConvTranspose_memory(module, inp, out)
    elif isinstance(module, nn.BatchNorm1d):
        return compute_BatchNorm1d_memory(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_memory(module, inp, out)
    elif isinstance(module, nn.BatchNorm3d):
        return compute_BatchNorm3d_memory(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_memory(module, inp, out)
    elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        return compute_Pool_memory(module, inp, out)
    elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)):
        return compute_AdaptivePool_memory(module, inp, out)
    elif isinstance(module, nn.Identity):
        return compute_Identity_memory(module, inp, out)
    elif isinstance(module, nn.Flatten):
        return compute_Flatten_memory(module, inp, out)
    elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
        return compute_Dropout_memory(module, inp, out)
    elif isinstance(module, nn.LayerNorm):
        return compute_LayerNorm_memory(module, inp, out)
    elif isinstance(module, nn.GroupNorm):
        return compute_GroupNorm_memory(module, inp, out)
    elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        return compute_InstanceNorm_memory(module, inp, out)
    elif isinstance(module, nn.LocalResponseNorm):
        return compute_LocalResponseNorm_memory(module, inp, out)
    elif isinstance(module, nn.Embedding):
        return compute_Embedding_memory(module, inp, out)
    elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        return compute_RNN_memory(module, inp, out)
    elif isinstance(module, nn.MultiheadAttention):
        return compute_MultiheadAttention_memory(module, inp, out)
    else:
        # Handle timm-specific and other custom layers by module class name
        module_name = type(module).__name__
        if module_name in ['LayerScale', 'LayerScale2d']:
            return compute_LayerScale_memory(module, inp, out)
        elif module_name == 'Flat':
            return compute_Flat_memory(module, inp, out)
        elif module_name in ['DropPath', 'StochasticDepth']:
            return compute_DropPath_memory(module, inp, out)
        elif module_name in ['FastAdaptiveAvgPool', 'FastGlobalAvgPool2d']:
            return compute_FastAdaptiveAvgPool_memory(module, inp, out)
        elif 'Scale' in module_name or 'Attention' in module_name:
            # Generic handling for scale-like or attention-like operations
            return compute_Generic_Scale_memory(module, inp, out)
        else:
            print(f"[Memory]: {module_name} is not supported!")
            return (0, 0)
    pass


def num_params(module):
    """Return the total number of elements in parameters and buffers.

    Parameters include all learnable parameters that require gradients while
    buffers hold running statistics (e.g., BatchNorm running mean/var) which are
    not part of the model's gradients but still occupy memory.  Both should be
    counted when estimating memory requirements.
    """
    param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    buffer_count = sum(b.numel() for b in module.buffers())
    return param_count + buffer_count


def compute_ReLU_memory(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU))
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * inp.size()[1:].numel()

    return (mread, mwrite)


def compute_PReLU_memory(module, inp, out):
    assert isinstance(module, (nn.PReLU))
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel() + num_params(module)
    mwrite = batch_size * inp.size()[1:].numel()

    return (mread, mwrite)


def compute_Conv2d_memory(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    out_c, out_h, out_w = out.size()[1:]

    # This includes weights with bias if the module contains it.
    # Parameters are shared across batch, so they should not be multiplied by batch_size
    mread = batch_size * inp.size()[1:].numel() + num_params(module)
    mwrite = batch_size * out_c * out_h * out_w
    return (mread, mwrite)


def compute_BatchNorm2d_memory(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    batch_size = inp.size()[0]

    # Read: input tensor + BatchNorm parameters and buffers (not multiplied by batch_size)
    param_size = num_params(module)
    mread = batch_size * inp.size()[1:].numel() + param_size
    mwrite = batch_size * out.size()[1:].numel()
    return (mread, mwrite)


def compute_Linear_memory(module, inp, out):
    assert isinstance(module, nn.Linear)
    # Linear layer can handle any input shape as long as last dim matches in_features
    assert inp.size(-1) == module.in_features
    assert out.size(-1) == module.out_features
    
    # Calculate memory based on actual tensor sizes
    mread = inp.numel() + num_params(module)
    mwrite = out.numel()

    return (mread, mwrite)


def compute_Identity_memory(module, inp, out):
    # Identity layer just passes data through
    assert isinstance(module, nn.Identity)
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    return (mread, mwrite)


def compute_Flatten_memory(module, inp, out):
    # Flatten just reshapes, no additional memory needed
    assert isinstance(module, nn.Flatten)
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    return (mread, mwrite)


def compute_Activation_memory(module, inp, out):
    """Generic memory computation for most activation functions"""
    # Most activation functions are element-wise operations that don't require additional parameters
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    return (mread, mwrite)


def compute_ELU_memory(module, inp, out):
    """Memory computation for ELU variants like SELU, CELU"""
    # These may have small parameter overhead but are mostly element-wise
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    
    # SELU and CELU might have small constant parameters but negligible memory impact
    if hasattr(module, 'alpha'):
        param_size = 1  # single parameter
    else:
        param_size = 0
    
    mread += param_size
    mwrite = batch_size * out.size()[1:].numel()
    return (mread, mwrite)


# New memory computation functions
def compute_Conv1d_memory(module, inp, out):
    """Memory computation for Conv1d"""
    assert isinstance(module, nn.Conv1d)
    assert len(inp.size()) == 3 and len(out.size()) == 3
    
    batch_size = inp.size()[0]
    # Parameters are shared across batch, so they should not be multiplied by batch_size
    mread = batch_size * inp.size()[1:].numel() + num_params(module)
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_Conv3d_memory(module, inp, out):
    """Memory computation for Conv3d"""
    assert isinstance(module, nn.Conv3d)
    assert len(inp.size()) == 5 and len(out.size()) == 5
    
    batch_size = inp.size()[0]
    # Parameters are shared across batch, so they should not be multiplied by batch_size
    mread = batch_size * inp.size()[1:].numel() + num_params(module)
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_ConvTranspose_memory(module, inp, out):
    """Memory computation for ConvTranspose layers"""
    batch_size = inp.size()[0]
    # Parameters are shared across batch, so they should not be multiplied by batch_size
    mread = batch_size * inp.size()[1:].numel() + num_params(module)
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_BatchNorm1d_memory(module, inp, out):
    """Memory computation for BatchNorm1d"""
    assert isinstance(module, nn.BatchNorm1d)
    batch_size = inp.size()[0]
    param_size = num_params(module)
    mread = batch_size * inp.size()[1:].numel() + param_size
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_BatchNorm3d_memory(module, inp, out):
    """Memory computation for BatchNorm3d"""
    assert isinstance(module, nn.BatchNorm3d)
    batch_size = inp.size()[0]
    param_size = num_params(module)
    
    mread = batch_size * inp.size()[1:].numel() + param_size
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_Pool_memory(module, inp, out):
    """Memory computation for pooling operations"""
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_AdaptivePool_memory(module, inp, out):
    """Memory computation for adaptive pooling operations"""
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_Dropout_memory(module, inp, out):
    """Memory computation for Dropout"""
    # Dropout passes data through without modification during inference
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_LayerNorm_memory(module, inp, out):
    """Memory computation for LayerNorm"""
    assert isinstance(module, nn.LayerNorm)
    batch_size = inp.size()[0]
    
    # LayerNorm needs to read input and parameters
    mread = batch_size * inp.size()[1:].numel() + num_params(module)
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_GroupNorm_memory(module, inp, out):
    """Memory computation for GroupNorm"""
    assert isinstance(module, nn.GroupNorm)
    batch_size = inp.size()[0]
    
    mread = batch_size * inp.size()[1:].numel() + num_params(module)
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_InstanceNorm_memory(module, inp, out):
    """Memory computation for InstanceNorm"""
    batch_size = inp.size()[0]
    
    mread = batch_size * inp.size()[1:].numel() + num_params(module)
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_LocalResponseNorm_memory(module, inp, out):
    """Memory computation for LocalResponseNorm"""
    assert isinstance(module, nn.LocalResponseNorm)
    batch_size = inp.size()[0]
    
    # LRN needs to access neighboring channels
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_Embedding_memory(module, inp, out):
    """Memory computation for Embedding"""
    assert isinstance(module, nn.Embedding)
    
    # Embedding reads from embedding table and indices
    mread = inp.numel() + num_params(module)  # indices + embedding table
    mwrite = out.numel()
    
    return (mread, mwrite)


def compute_RNN_memory(module, inp, out):
    """Memory computation for RNN layers"""
    if len(inp.size()) == 3:
        batch_size, seq_len, input_size = inp.size()
    else:
        batch_size, input_size = inp.size()
        seq_len = 1
    
    hidden_size = module.hidden_size
    num_layers = module.num_layers

    # RNN needs to read input, parameters, and hidden states
    param_size = num_params(module)
    hidden_state_size = batch_size * num_layers * hidden_size

    if getattr(module, 'bidirectional', False):
        hidden_state_size *= 2

    if isinstance(module, nn.LSTM):
        hidden_state_size *= 2  # cell state + hidden state

    mread = batch_size * seq_len * input_size + param_size + hidden_state_size

    # Output size - handle tuple/list outputs
    out_tensor = out[0] if isinstance(out, (tuple, list)) else out
    if len(out_tensor.size()) == 3:
        mwrite = batch_size * seq_len * out_tensor.size(2)
    else:
        mwrite = out_tensor.numel()
    
    return (mread, mwrite)


def compute_MultiheadAttention_memory(module, inp, out):
    """Memory computation for MultiheadAttention"""
    if isinstance(inp, (list, tuple)):
        query = inp[0]
    else:
        query = inp
        
    if len(query.size()) == 3:
        batch_size, seq_len, embed_dim = query.size()
    else:
        batch_size, embed_dim = query.size()
        seq_len = 1
    
    # Attention needs to read Q, K, V and parameters
    # Simplified estimation
    mread = batch_size * seq_len * embed_dim * 3 + num_params(module)  # Q, K, V
    mwrite = batch_size * seq_len * embed_dim  # output
    
    return (mread, mwrite)


# TIMM-specific and custom layer implementations for Memory
def compute_LayerScale_memory(module, inp, out):
    """Memory computation for LayerScale/LayerScale2d"""
    batch_size = inp.size()[0]
    
    # LayerScale reads input and scale parameters
    mread = batch_size * inp.size()[1:].numel() + num_params(module)
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_Flat_memory(module, inp, out):
    """Memory computation for Flat operation"""
    batch_size = inp.size()[0]
    
    # Flat is just reshaping, input and output are the same data
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_DropPath_memory(module, inp, out):
    """Memory computation for DropPath/StochasticDepth"""
    batch_size = inp.size()[0]
    
    # DropPath passes through or zeros out during training
    # During inference, it's pass-through
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_FastAdaptiveAvgPool_memory(module, inp, out):
    """Compute memory for FastAdaptiveAvgPool/FastGlobalAvgPool2d from TIMM"""
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    return (mread, mwrite)


def compute_Generic_Scale_memory(module, inp, out):
    """Generic memory computation for scale-like operations"""
    batch_size = inp.size()[0]
    
    # Generic scale operation reads input and parameters
    param_size = num_params(module) if hasattr(module, 'parameters') else 0
    mread = batch_size * inp.size()[1:].numel() + param_size
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)


def compute_Softmax_memory(module, inp, out):
    """Memory computation for Softmax"""
    assert isinstance(module, nn.Softmax)
    batch_size = inp.size()[0]
    
    # Softmax is an element-wise operation with no additional parameters
    # During computation, it needs to read input and potentially store intermediate results
    # For numerical stability, softmax typically subtracts the max value first
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    
    return (mread, mwrite)
