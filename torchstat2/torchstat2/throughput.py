#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
from typing import Union, List, Tuple, Optional


def throughput(model: nn.Module, 
               input_shape: Union[Tuple, List] = (3, 224, 224),
               device: str = 'cuda',
               batch_sizes: Union[int, List[int]] = [1, 8, 16, 32],
               warmup_time: float = 2.0,
               test_time: float = 10.0,
               model_name: str = "Model") -> dict:
    """
    Measure model throughput on specified device
    
    Args:
        model: PyTorch model to test
        input_shape: Input tensor shape (C, H, W) or (seq_len, features)
        device: Device to run on ('cuda', 'cpu', or specific like 'cuda:0')
        batch_sizes: Batch size(s) to test, can be int or list of ints
        warmup_time: Warmup time in seconds
        test_time: Test time in seconds for measurement
        model_name: Name for display purposes
        
    Returns:
        Dictionary with throughput results
    """
    # Ensure batch_sizes is a list
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    
    # Check device availability
    if device.startswith('cuda') and not torch.cuda.is_available():
        print(f"CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    results = {
        'model_name': model_name,
        'device': device,
        'input_shape': input_shape,
        'measurements': []
    }
    
    print(f"\n{'='*60}")
    print(f"Throughput Analysis: {model_name}")
    print(f"Device: {device}")
    print(f"Input shape: {input_shape}")
    print(f"{'='*60}")
    
    for batch_size in batch_sizes:
        try:
            throughput_result = _measure_single_batch(
                model, input_shape, device, batch_size, 
                warmup_time, test_time
            )
            results['measurements'].append(throughput_result)
            
            print(f"Batch size {batch_size:3d}: {throughput_result['throughput']:.2f} images/s "
                  f"(avg: {throughput_result['avg_latency']*1000:.2f}ms, "
                  f"std: {throughput_result['std_latency']*1000:.2f}ms)")
                  
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Batch size {batch_size:3d}: Out of memory")
                # Clear cache and continue with next batch size
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
        except Exception as e:
            print(f"Batch size {batch_size:3d}: Error - {e}")
            continue
    
    # Find optimal batch size
    if results['measurements']:
        optimal = max(results['measurements'], key=lambda x: x['throughput'])
        results['optimal_batch_size'] = optimal['batch_size']
        results['max_throughput'] = optimal['throughput']
        
        print(f"\nOptimal batch size: {optimal['batch_size']} "
              f"(max throughput: {optimal['throughput']:.2f} images/s)")
    
    return results


def _measure_single_batch(model: nn.Module,
                         input_shape: Tuple,
                         device: str,
                         batch_size: int,
                         warmup_time: float,
                         test_time: float) -> dict:
    """Measure throughput for a single batch size"""
    
    # Create input tensor
    if len(input_shape) == 3:  # Image data (C, H, W)
        inputs = torch.randn(batch_size, *input_shape, device=device)
    elif len(input_shape) == 2:  # Sequence data (seq_len, features)
        inputs = torch.randn(batch_size, *input_shape, device=device)
    elif len(input_shape) == 1:  # 1D data (features,)
        inputs = torch.randn(batch_size, *input_shape, device=device)
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")
    
    # Clear GPU cache
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Efficient warmup phase
    model.eval()
    warmup_iters = max(5, int(warmup_time * 20))  # Adaptive warmup iterations
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(inputs)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
    
    # Measurement phase with fixed iterations for stability
    timing = []
    min_iterations = 30
    max_iterations = 500
    target_iterations = max(min_iterations, min(max_iterations, int(test_time * 50)))
    
    with torch.no_grad():
        for i in range(target_iterations):
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(inputs)
            
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            
            timing.append(time.time() - start)
            
            # Early termination if we have enough stable measurements
            if i >= min_iterations:
                current_time = time.time()
                if (current_time - start) > test_time:
                    break
    
    # Calculate statistics with outlier removal for stability
    timing_tensor = torch.tensor(timing, dtype=torch.float32)
    
    # Remove outliers if we have enough samples
    if len(timing) > 10:
        sorted_timing = torch.sort(timing_tensor)[0]
        trim_count = max(1, len(timing) // 20)  # Remove top/bottom 5%
        timing_tensor = sorted_timing[trim_count:-trim_count]
    
    avg_latency = timing_tensor.mean().item()
    std_latency = timing_tensor.std().item()
    throughput_value = batch_size / avg_latency
    
    return {
        'batch_size': batch_size,
        'throughput': throughput_value,  # images/s
        'avg_latency': avg_latency,      # seconds
        'std_latency': std_latency,      # seconds
        'num_measurements': len(timing)
    }


def compare_models_throughput(models_dict: dict,
                            input_shape: Tuple = (3, 224, 224),
                            device: str = 'cuda',
                            batch_sizes: List[int] = [1, 8, 16, 32],
                            **kwargs) -> dict:
    """
    Compare throughput across multiple models
    
    Args:
        models_dict: Dictionary of {model_name: model}
        input_shape: Input tensor shape
        device: Device to run on
        batch_sizes: Batch sizes to test
        **kwargs: Additional arguments for throughput function
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    print(f"\n{'='*80}")
    print(f"Model Throughput Comparison")
    print(f"{'='*80}")
    
    for model_name, model in models_dict.items():
        print(f"\nTesting {model_name}...")
        results[model_name] = throughput(
            model=model,
            input_shape=input_shape,
            device=device,
            batch_sizes=batch_sizes,
            model_name=model_name,
            **kwargs
        )
    
    # Summary comparison
    print(f"\n{'='*80}")
    print(f"Throughput Comparison Summary")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Optimal Batch':<12} {'Max Throughput':<15} {'Device':<10}")
    print(f"{'-'*60}")
    
    for model_name, result in results.items():
        if result['measurements']:
            print(f"{model_name:<20} {result['optimal_batch_size']:<12} "
                  f"{result['max_throughput']:<15.2f} {result['device']:<10}")
        else:
            print(f"{model_name:<20} {'N/A':<12} {'N/A':<15} {result['device']:<10}")
    
    return results


def analyze_batch_scaling(model: nn.Module,
                         input_shape: Tuple = (3, 224, 224),
                         device: str = 'cuda',
                         max_batch_size: int = 128,
                         model_name: str = "Model") -> dict:
    """
    Analyze how throughput scales with batch size
    
    Args:
        model: PyTorch model to test
        input_shape: Input tensor shape
        device: Device to run on
        max_batch_size: Maximum batch size to test
        model_name: Name for display purposes
        
    Returns:
        Dictionary with scaling analysis results
    """
    # Generate batch sizes to test (powers of 2 + some in-between values)
    batch_sizes = []
    batch_size = 1
    while batch_size <= max_batch_size:
        batch_sizes.append(batch_size)
        if batch_size < 16:
            batch_size *= 2
        else:
            batch_size += 16
    
    # Filter out batch sizes that exceed max_batch_size
    batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
    
    print(f"\nAnalyzing batch size scaling for {model_name}")
    print(f"Testing batch sizes: {batch_sizes}")
    
    results = throughput(
        model=model,
        input_shape=input_shape,
        device=device,
        batch_sizes=batch_sizes,
        model_name=model_name,
        warmup_time=1.0,  # Shorter for scaling analysis
        test_time=5.0
    )
    
    # Analyze scaling efficiency
    if len(results['measurements']) >= 2:
        baseline = results['measurements'][0]['throughput']  # batch_size=1 throughput
        
        print(f"\nScaling Analysis:")
        print(f"{'Batch Size':<10} {'Throughput':<12} {'Scaling Factor':<15} {'Efficiency':<12}")
        print(f"{'-'*50}")
        
        for measurement in results['measurements']:
            batch_size = measurement['batch_size']
            throughput_val = measurement['throughput']
            scaling_factor = throughput_val / baseline
            efficiency = scaling_factor / batch_size * 100  # Percentage of linear scaling
            
            print(f"{batch_size:<10} {throughput_val:<12.2f} "
                  f"{scaling_factor:<15.2f} {efficiency:<12.1f}%")
    
    return results


# Convenience function for integration with existing torchstat2
def stat_with_throughput(model: nn.Module,
                        input_size: Tuple,
                        device: str = 'cuda',
                        batch_sizes: List[int] = [1, 8, 16],
                        query_granularity: int = 1):
    """
    Combined function that runs both torchstat2 analysis and throughput measurement
    
    Args:
        model: PyTorch model to analyze
        input_size: Input tensor size (without batch dimension)
        device: Device for throughput testing
        batch_sizes: Batch sizes to test for throughput
        query_granularity: Granularity for torchstat2 analysis
    """
    from .statistics import stat
    
    # Store original model device and state
    original_device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
    
    # For torchstat2 analysis, ensure model is on CPU (torchstat2 expects CPU)
    print("Running TorchStat Analysis...")
    print("=" * 60)
    model_cpu = model.cpu()
    stat(model_cpu, input_size, query_granularity)
    
    print("\nRunning Throughput Analysis...")
    print("=" * 60)
    
    # Move model to specified device for throughput testing
    if device.startswith('cuda') and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
        device = 'cpu'
    
    throughput_results = throughput(
        model=model,
        input_shape=input_size,
        device=device,
        batch_sizes=batch_sizes,
        model_name="Model"
    )
    
    # Restore model to original device
    model = model.to(original_device)
    
    return throughput_results
