"""
Custom optimized kernels for JAX operations
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


@jit
def optimized_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Optimized matrix multiplication with JIT compilation

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Matrix product
    """
    return jnp.dot(a, b)


@jit
def fused_operations(x: jnp.ndarray, weights: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """
    Fused linear transformation + activation

    Combines multiple operations to reduce kernel launch overhead

    Args:
        x: Input array
        weights: Weight matrix
        bias: Bias vector

    Returns:
        Activated output
    """
    return jax.nn.relu(jnp.dot(x, weights) + bias)
