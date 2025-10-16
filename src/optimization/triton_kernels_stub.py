"""
Stub implementations for Triton kernels when Triton is not available.
"""

import torch
import math
from typing import Tuple, Optional


def fused_matmul_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fallback implementation without Triton optimization."""
    return torch.relu(a @ b)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """Fallback implementation without Triton optimization."""
    B, H, M, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Standard attention (not memory efficient, but works)
    scores = torch.einsum('bhmd,bhnd->bhmn', q, k) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum('bhmn,bhnd->bhmd', attn, v)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fallback implementation without Triton optimization."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


def compute_inventory_policy(
    demand: torch.Tensor,
    stock: torch.Tensor,
    lead_time: torch.Tensor,
    service_level: float = 0.95,
    holding_cost: float = 1.0,
    order_cost: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback implementation without Triton optimization."""
    N, T = demand.shape

    # Compute statistics on CPU (more compatible)
    mean_demand = torch.mean(demand, dim=1)
    std_demand = torch.std(demand, dim=1)

    # Lead time demand
    ltd_mean = mean_demand * lead_time
    ltd_std = std_demand * torch.sqrt(lead_time)

    # Z-score for service level
    z_score = 1.645 + (service_level - 0.95) * 17.075

    # Reorder point
    safety_stock = z_score * ltd_std
    reorder_point = ltd_mean + safety_stock

    # EOQ
    annual_demand = mean_demand * 365.0
    eoq = torch.sqrt(2.0 * annual_demand * order_cost / holding_cost)
    order_qty = torch.maximum(eoq, mean_demand * 7.0)

    return order_qty, reorder_point


def benchmark_kernel(func, *args, warmup: int = 10, iterations: int = 100, **kwargs) -> dict:
    """Fallback benchmarking implementation."""
    import time

    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000 / iterations

    return {
        'mean_ms': elapsed_ms,
        'iterations': iterations,
        'throughput': 1000 / elapsed_ms,
    }
