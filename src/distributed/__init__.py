"""
Distributed Training Framework for JAX Inventory Optimizer

This module provides utilities for distributed training across multiple GPUs/TPUs
using JAX's pmap and pjit primitives.
"""

from .trainer import DistributedTrainer
from .data_parallel import DataParallelTrainer
from .model_parallel import ModelParallelTrainer
from .utils import (
    setup_distributed,
    get_device_mesh,
    shard_data,
    gather_metrics,
)

__all__ = [
    "DistributedTrainer",
    "DataParallelTrainer",
    "ModelParallelTrainer",
    "setup_distributed",
    "get_device_mesh",
    "shard_data",
    "gather_metrics",
]
