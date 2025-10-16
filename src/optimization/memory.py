"""
Memory optimization utilities for JAX
"""

import jax
import jax.numpy as jnp
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Utilities for optimizing GPU memory usage"""

    @staticmethod
    def check_memory_usage() -> Dict[str, float]:
        """
        Check current GPU memory usage

        Returns:
            Dictionary with memory statistics in MB
        """
        try:
            devices = jax.devices()
            memory_stats = {}

            for i, device in enumerate(devices):
                if hasattr(device, 'memory_stats'):
                    stats = device.memory_stats()
                    memory_stats[f'device_{i}'] = {
                        'bytes_in_use_mb': stats.get('bytes_in_use', 0) / (1024**2),
                        'peak_bytes_in_use_mb': stats.get('peak_bytes_in_use', 0) / (1024**2),
                    }

            return memory_stats
        except Exception as e:
            logger.warning(f"Could not get memory stats: {e}")
            return {}

    @staticmethod
    def clear_caches():
        """Clear JAX compilation caches"""
        jax.clear_caches()
        logger.info("Cleared JAX compilation caches")


def check_memory_usage() -> Dict[str, float]:
    """Helper function to check memory usage"""
    return MemoryOptimizer.check_memory_usage()
