"""
Distributed training utilities for JAX
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def setup_distributed() -> Dict[str, Any]:
    """
    Setup distributed training environment

    Returns:
        dict: Configuration including device count, mesh, etc.
    """
    device_count = jax.device_count()
    local_device_count = jax.local_device_count()

    logger.info(f"Total devices: {device_count}")
    logger.info(f"Local devices: {local_device_count}")
    logger.info(f"Device type: {jax.devices()[0].platform}")

    return {
        "device_count": device_count,
        "local_device_count": local_device_count,
        "devices": jax.devices(),
        "platform": jax.devices()[0].platform,
    }


def get_device_mesh(
    device_count: Optional[int] = None,
    mesh_shape: Optional[Tuple[int, ...]] = None
) -> Mesh:
    """
    Create device mesh for parallel computation

    Args:
        device_count: Number of devices to use (None = all)
        mesh_shape: Shape of device mesh (e.g., (2, 4) for 8 devices)

    Returns:
        Mesh: JAX mesh object
    """
    if device_count is None:
        device_count = jax.device_count()

    devices = jax.devices()[:device_count]

    if mesh_shape is None:
        # Default: 1D mesh for data parallelism
        mesh_shape = (device_count,)

    device_array = mesh_utils.create_device_mesh(mesh_shape, devices)

    # Create mesh with axis names
    if len(mesh_shape) == 1:
        axis_names = ('data',)
    elif len(mesh_shape) == 2:
        axis_names = ('data', 'model')
    else:
        axis_names = tuple(f'axis_{i}' for i in range(len(mesh_shape)))

    mesh = Mesh(device_array, axis_names)
    logger.info(f"Created mesh with shape {mesh_shape} and axes {axis_names}")

    return mesh


def shard_data(
    data: jnp.ndarray,
    mesh: Mesh,
    partition_spec: P
) -> jnp.ndarray:
    """
    Shard data across devices according to partition spec

    Args:
        data: Input data array
        mesh: Device mesh
        partition_spec: Partitioning specification

    Returns:
        Sharded array
    """
    from jax.experimental.shard_map import shard_map

    # Use device_put with sharding
    from jax.sharding import NamedSharding
    sharding = NamedSharding(mesh, partition_spec)

    return jax.device_put(data, sharding)


def gather_metrics(
    local_metrics: Dict[str, jnp.ndarray],
    reduce_fn: str = "mean"
) -> Dict[str, float]:
    """
    Gather and reduce metrics from all devices

    Args:
        local_metrics: Metrics from local device
        reduce_fn: Reduction function ('mean', 'sum', 'min', 'max')

    Returns:
        Reduced metrics as Python scalars
    """
    reduced_metrics = {}

    for key, value in local_metrics.items():
        if reduce_fn == "mean":
            reduced = jnp.mean(value)
        elif reduce_fn == "sum":
            reduced = jnp.sum(value)
        elif reduce_fn == "min":
            reduced = jnp.min(value)
        elif reduce_fn == "max":
            reduced = jnp.max(value)
        else:
            raise ValueError(f"Unknown reduce_fn: {reduce_fn}")

        # Convert to Python scalar
        reduced_metrics[key] = float(reduced)

    return reduced_metrics


def synchronize_gradients(gradients: Any) -> Any:
    """
    Synchronize gradients across devices using all-reduce

    Args:
        gradients: Gradient pytree

    Returns:
        Synchronized gradients
    """
    from jax.lax import pmean

    # Average gradients across all devices
    return jax.tree.map(lambda g: pmean(g, axis_name='data'), gradients)


def create_data_loader_parallel(
    data: jnp.ndarray,
    batch_size: int,
    num_devices: int,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[jnp.ndarray, int]:
    """
    Create parallel data loader for distributed training

    Args:
        data: Full dataset
        batch_size: Global batch size (will be divided across devices)
        num_devices: Number of devices
        shuffle: Whether to shuffle data
        seed: Random seed for shuffling

    Returns:
        Tuple of (sharded_data, num_batches)
    """
    n_samples = data.shape[0]

    # Ensure batch_size is divisible by num_devices
    per_device_batch_size = batch_size // num_devices
    assert batch_size % num_devices == 0, \
        f"batch_size {batch_size} must be divisible by num_devices {num_devices}"

    if shuffle:
        key = jax.random.PRNGKey(seed)
        perm = jax.random.permutation(key, n_samples)
        data = data[perm]

    # Trim data to make it divisible by batch_size
    num_batches = n_samples // batch_size
    data = data[:num_batches * batch_size]

    # Reshape to [num_batches, num_devices, per_device_batch_size, ...]
    data = data.reshape(
        num_batches,
        num_devices,
        per_device_batch_size,
        *data.shape[1:]
    )

    return data, num_batches


def all_gather(x: jnp.ndarray, axis_name: str = 'data') -> jnp.ndarray:
    """
    Gather array from all devices

    Args:
        x: Local array
        axis_name: Name of parallel axis

    Returns:
        Gathered array concatenated along first dimension
    """
    from jax.lax import all_gather as jax_all_gather
    return jax_all_gather(x, axis_name=axis_name)


def log_distributed_info():
    """Log information about distributed setup"""
    info = setup_distributed()

    logger.info("=" * 60)
    logger.info("Distributed Training Setup")
    logger.info("=" * 60)
    logger.info(f"Platform: {info['platform']}")
    logger.info(f"Total devices: {info['device_count']}")
    logger.info(f"Local devices: {info['local_device_count']}")

    for i, device in enumerate(info['devices']):
        logger.info(f"Device {i}: {device}")

    logger.info("=" * 60)


if __name__ == "__main__":
    # Test distributed setup
    logging.basicConfig(level=logging.INFO)
    log_distributed_info()
