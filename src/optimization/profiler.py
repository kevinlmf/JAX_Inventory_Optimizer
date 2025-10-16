"""
GPU Profiling utilities for JAX operations

Provides tools to profile and optimize GPU performance
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import time
import functools
from typing import Callable, Dict, Any, Optional, List
import logging
from contextlib import contextmanager
import json

logger = logging.getLogger(__name__)


class GPUProfiler:
    """
    GPU Performance Profiler for JAX operations

    Measures:
    - Execution time
    - Memory usage
    - GPU utilization
    - Kernel launch overhead
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize profiler

        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.measurements = []
        self.current_context = None
        self._start_time = None
        self._start_memory = None

    def __enter__(self):
        """Enter context manager"""
        if self.enabled:
            # Block until all previous computations complete
            jax.block_until_ready(jnp.zeros(1))
            self._start_time = time.perf_counter()
            self._start_memory = self._get_gpu_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        if self.enabled and self._start_time is not None:
            # Block until current computation completes
            jax.block_until_ready(jnp.zeros(1))

            end_time = time.perf_counter()
            end_memory = self._get_gpu_memory()

            measurement = {
                'name': 'profiled_block',
                'duration_ms': (end_time - self._start_time) * 1000,
                'memory_mb': end_memory - self._start_memory,
                'timestamp': time.time(),
            }

            self.measurements.append(measurement)
        return False

    @contextmanager
    def profile(self, name: str, **metadata):
        """
        Context manager for profiling a code block

        Args:
            name: Name of the operation being profiled
            **metadata: Additional metadata to store

        Example:
            with profiler.profile("matrix_multiply"):
                result = jnp.dot(a, b)
        """
        if not self.enabled:
            yield
            return

        # Block until all previous computations complete
        jax.block_until_ready(jnp.zeros(1))

        start_time = time.perf_counter()
        start_memory = self._get_gpu_memory()

        try:
            yield
        finally:
            # Block until current computation completes
            jax.block_until_ready(jnp.zeros(1))

            end_time = time.perf_counter()
            end_memory = self._get_gpu_memory()

            measurement = {
                'name': name,
                'duration_ms': (end_time - start_time) * 1000,
                'memory_mb': end_memory - start_memory,
                'timestamp': time.time(),
                **metadata
            }

            self.measurements.append(measurement)
            logger.debug(
                f"Profile [{name}]: {measurement['duration_ms']:.2f}ms, "
                f"Memory: {measurement['memory_mb']:.2f}MB"
            )

    def _get_gpu_memory(self) -> float:
        """
        Get current GPU memory usage in MB

        Returns:
            Memory usage in megabytes
        """
        try:
            # Try to get device memory usage
            devices = jax.devices()
            if devices and hasattr(devices[0], 'memory_stats'):
                stats = devices[0].memory_stats()
                return stats.get('bytes_in_use', 0) / (1024 ** 2)
        except Exception as e:
            logger.debug(f"Could not get GPU memory: {e}")

        return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all measurements

        Returns:
            Dictionary with summary statistics
        """
        if not self.measurements:
            return {}

        durations = [m['duration_ms'] for m in self.measurements]
        memories = [m['memory_mb'] for m in self.measurements]

        return {
            'total_operations': len(self.measurements),
            'total_time_ms': sum(durations),
            'avg_time_ms': sum(durations) / len(durations),
            'max_time_ms': max(durations),
            'min_time_ms': min(durations),
            'total_memory_mb': sum(memories),
            'avg_memory_mb': sum(memories) / len(memories) if memories else 0,
        }

    def get_measurements(self) -> List[Dict[str, Any]]:
        """Get all measurements"""
        return self.measurements

    def clear(self):
        """Clear all measurements"""
        self.measurements = []

    def save_to_file(self, filepath: str):
        """
        Save measurements to JSON file

        Args:
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump({
                'measurements': self.measurements,
                'summary': self.get_summary()
            }, f, indent=2)

        logger.info(f"Saved profile data to {filepath}")

    def print_summary(self):
        """Print formatted summary of measurements"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("GPU Profiling Summary")
        print("=" * 60)

        for key, value in summary.items():
            if 'time' in key:
                print(f"{key:30s}: {value:>10.2f} ms")
            elif 'memory' in key:
                print(f"{key:30s}: {value:>10.2f} MB")
            else:
                print(f"{key:30s}: {value:>10}")

        print("=" * 60 + "\n")

        # Print top 5 slowest operations
        if self.measurements:
            print("Top 5 Slowest Operations:")
            print("-" * 60)
            sorted_ops = sorted(
                self.measurements,
                key=lambda x: x['duration_ms'],
                reverse=True
            )

            for i, op in enumerate(sorted_ops[:5], 1):
                print(f"{i}. {op['name']:30s}: {op['duration_ms']:>10.2f} ms")

            print("=" * 60 + "\n")


def profile_function(name: Optional[str] = None, enabled: bool = True):
    """
    Decorator to profile a function

    Args:
        name: Custom name for the operation (default: function name)
        enabled: Whether profiling is enabled

    Example:
        @profile_function(name="my_operation")
        def my_func(x):
            return jnp.dot(x, x.T)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            op_name = name or func.__name__

            # Block before timing
            jax.block_until_ready(jnp.zeros(1))
            start = time.perf_counter()

            result = func(*args, **kwargs)

            # Block until result is ready
            jax.block_until_ready(result)
            duration = (time.perf_counter() - start) * 1000

            logger.info(f"Function [{op_name}] took {duration:.2f}ms")

            return result

        return wrapper
    return decorator


def benchmark(
    func: Callable,
    *args,
    num_runs: int = 100,
    warmup_runs: int = 10,
    name: Optional[str] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark a function with multiple runs

    Args:
        func: Function to benchmark
        *args: Positional arguments for function
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        name: Custom name for operation
        **kwargs: Keyword arguments for function

    Returns:
        Dictionary with timing statistics
    """
    op_name = name or func.__name__

    logger.info(f"Benchmarking [{op_name}] with {num_runs} runs...")

    # Warmup
    for _ in range(warmup_runs):
        result = func(*args, **kwargs)
        jax.block_until_ready(result)

    # Benchmark
    times = []
    for _ in range(num_runs):
        jax.block_until_ready(jnp.zeros(1))
        start = time.perf_counter()

        result = func(*args, **kwargs)
        jax.block_until_ready(result)

        duration = (time.perf_counter() - start) * 1000
        times.append(duration)

    # Calculate statistics
    times_sorted = sorted(times)
    stats = {
        'name': op_name,
        'num_runs': num_runs,
        'mean_ms': sum(times) / len(times),
        'median_ms': times_sorted[len(times) // 2],
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': jnp.std(jnp.array(times)),
        'p95_ms': times_sorted[int(len(times) * 0.95)],
        'p99_ms': times_sorted[int(len(times) * 0.99)],
    }

    logger.info(
        f"Benchmark [{op_name}]: "
        f"mean={stats['mean_ms']:.2f}ms, "
        f"median={stats['median_ms']:.2f}ms, "
        f"p95={stats['p95_ms']:.2f}ms"
    )

    return stats


class CompilationTracker:
    """
    Track JAX compilation events

    Helps identify which functions are being recompiled
    and measure compilation overhead
    """

    def __init__(self):
        self.compilation_count = 0
        self.compilation_times = []

    def track_jit(self, func: Callable, name: Optional[str] = None) -> Callable:
        """
        Wrap a function to track its JIT compilation

        Args:
            func: Function to track
            name: Custom name

        Returns:
            Wrapped function
        """
        op_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()

            # First call will trigger compilation
            result = func(*args, **kwargs)
            jax.block_until_ready(result)

            compile_time = (time.perf_counter() - start) * 1000

            self.compilation_count += 1
            self.compilation_times.append({
                'name': op_name,
                'time_ms': compile_time
            })

            logger.info(
                f"Compiled [{op_name}] in {compile_time:.2f}ms "
                f"(compilation #{self.compilation_count})"
            )

            return result

        return wrapper

    def get_summary(self) -> Dict[str, Any]:
        """Get compilation summary"""
        if not self.compilation_times:
            return {}

        times = [c['time_ms'] for c in self.compilation_times]

        return {
            'total_compilations': self.compilation_count,
            'total_compile_time_ms': sum(times),
            'avg_compile_time_ms': sum(times) / len(times),
            'compilations': self.compilation_times
        }


if __name__ == "__main__":
    # Test profiler
    logging.basicConfig(level=logging.INFO)

    profiler = GPUProfiler()

    # Test operations
    with profiler.profile("matrix_multiply"):
        a = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))
        b = jax.random.normal(jax.random.PRNGKey(1), (1000, 1000))
        c = jnp.dot(a, b)
        jax.block_until_ready(c)

    with profiler.profile("vmap_operation"):
        def square(x):
            return x ** 2

        data = jax.random.normal(jax.random.PRNGKey(2), (10000, 100))
        result = vmap(square)(data)
        jax.block_until_ready(result)

    # Print summary
    profiler.print_summary()

    # Benchmark example
    @jit
    def matmul(a, b):
        return jnp.dot(a, b)

    a = jax.random.normal(jax.random.PRNGKey(0), (512, 512))
    b = jax.random.normal(jax.random.PRNGKey(1), (512, 512))

    stats = benchmark(matmul, a, b, num_runs=100, name="JIT MatMul")
    print("\nBenchmark Results:")
    for key, value in stats.items():
        print(f"{key:15s}: {value}")
