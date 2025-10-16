"""
GPU Optimization and Profiling Utilities

This module provides tools for:
- GPU performance profiling
- Memory optimization
- Custom kernel implementations
- Performance benchmarking
"""

from .profiler import GPUProfiler, profile_function, benchmark
from .memory import MemoryOptimizer, check_memory_usage
from .kernels import optimized_matmul, fused_operations

# Try to import Triton kernels, fall back to stubs if not available
try:
    from .triton_kernels import (
        fused_matmul_relu,
        flash_attention,
        rms_norm,
        compute_inventory_policy,
        benchmark_kernel,
    )
    TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    from .triton_kernels_stub import (
        fused_matmul_relu,
        flash_attention,
        rms_norm,
        compute_inventory_policy,
        benchmark_kernel,
    )
    TRITON_AVAILABLE = False

__all__ = [
    "GPUProfiler",
    "profile_function",
    "benchmark",
    "MemoryOptimizer",
    "check_memory_usage",
    "optimized_matmul",
    "fused_operations",
    "fused_matmul_relu",
    "flash_attention",
    "rms_norm",
    "compute_inventory_policy",
    "benchmark_kernel",
    "TRITON_AVAILABLE",
]
