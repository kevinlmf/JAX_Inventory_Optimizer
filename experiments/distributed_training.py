"""
Distributed Training Experiment Script

This script demonstrates distributed training capabilities using JAX
with different parallelism strategies (auto, data, model, hybrid).

Usage:
    python experiments/distributed_training.py --strategy auto
    python experiments/distributed_training.py --strategy data --epochs 50
    python experiments/distributed_training.py --strategy auto --profile
"""

import sys
from pathlib import Path
import argparse
import logging
import time
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from src.distributed.trainer import DistributedTrainer
from src.distributed.utils import log_distributed_info, setup_distributed
from src.optimization.profiler import GPUProfiler
from src.methods.ml_methods.transformer import TimeSeriesTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def make_json_serializable(obj):
    """
    Convert JAX/numpy objects to JSON-serializable format

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    import jax

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__class__') and 'jax' in str(obj.__class__):
        # JAX Device objects
        return str(obj)
    elif isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj


def generate_synthetic_inventory_data(
    num_samples: int = 1000,
    seq_len: int = 50,
    input_dim: int = 1,
    seed: int = 42
) -> tuple:
    """
    Generate synthetic time series data for inventory demand forecasting

    Args:
        num_samples: Number of samples
        seq_len: Sequence length
        input_dim: Input dimension
        seed: Random seed

    Returns:
        Tuple of (train_data, eval_data) dictionaries
    """
    logger.info(f"Generating {num_samples} synthetic samples...")

    np.random.seed(seed)

    # Generate time series with seasonality + trend + noise
    t = np.arange(num_samples + seq_len)

    # Multiple seasonal components
    annual = 15 * np.sin(2 * np.pi * t / 365.25)
    weekly = 8 * np.sin(2 * np.pi * t / 7)
    monthly = 5 * np.sin(2 * np.pi * t / 30.44)
    trend = 0.01 * t
    noise = np.random.normal(0, 3, len(t))

    # Combine components
    demand = 60 + annual + weekly + monthly + trend + noise
    demand = np.maximum(demand, 0)  # Ensure non-negative

    # Create sequences
    X = []
    y = []
    for i in range(num_samples):
        X.append(demand[i:i + seq_len])
        y.append(demand[i + seq_len])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Add feature dimension
    if input_dim == 1:
        X = X.reshape(num_samples, seq_len, 1)
        y = y.reshape(num_samples, 1)

    # Split into train/eval (80/20)
    split_idx = int(0.8 * num_samples)

    train_data = {
        'input': X[:split_idx],
        'target': y[:split_idx]
    }

    eval_data = {
        'input': X[split_idx:],
        'target': y[split_idx:]
    }

    logger.info(f"Train samples: {len(train_data['input'])}")
    logger.info(f"Eval samples: {len(eval_data['input'])}")
    logger.info(f"Input shape: {train_data['input'].shape}")
    logger.info(f"Target shape: {train_data['target'].shape}")

    return train_data, eval_data


def create_simple_model(seq_len: int = 50, output_size: int = 1) -> nn.Module:
    """
    Create a simple model for distributed training experiments

    Args:
        seq_len: Sequence length
        output_size: Output size

    Returns:
        Flax neural network module
    """
    class SimpleRNN(nn.Module):
        hidden_size: int = 64
        output_size: int = 1

        @nn.compact
        def __call__(self, x, training=False):
            # Simple LSTM-like model
            batch_size, seq_len, input_dim = x.shape

            # Flatten sequence
            x = x.reshape(batch_size, seq_len * input_dim)

            # Dense layers
            x = nn.Dense(self.hidden_size)(x)
            x = nn.relu(x)
            x = nn.Dropout(0.1)(x, deterministic=not training)

            x = nn.Dense(self.hidden_size // 2)(x)
            x = nn.relu(x)
            x = nn.Dropout(0.1)(x, deterministic=not training)

            x = nn.Dense(self.output_size)(x)

            return x

    return SimpleRNN(output_size=output_size)


def create_transformer_model(
    seq_len: int = 50,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    output_size: int = 1
) -> nn.Module:
    """
    Create a Transformer model for distributed training

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        output_size: Output size

    Returns:
        TimeSeriesTransformer module
    """
    return TimeSeriesTransformer(
        seq_len=seq_len,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_model * 4,
        dropout_rate=0.1,
        output_size=output_size
    )


def run_distributed_training_experiment(
    strategy: str = 'auto',
    model_type: str = 'simple',
    num_samples: int = 1000,
    seq_len: int = 50,
    batch_size: int = 32,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    enable_profiling: bool = False,
    save_results: bool = True,
    seed: int = 42
):
    """
    Run distributed training experiment

    Args:
        strategy: Parallelism strategy ('auto', 'data', 'model', 'hybrid')
        model_type: Model type ('simple' or 'transformer')
        num_samples: Number of training samples
        seq_len: Sequence length
        batch_size: Global batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        enable_profiling: Enable GPU profiling
        save_results: Save results to file
        seed: Random seed
    """
    logger.info("=" * 80)
    logger.info("DISTRIBUTED TRAINING EXPERIMENT")
    logger.info("=" * 80)

    # Log distributed info
    log_distributed_info()

    # Initialize profiler
    profiler = GPUProfiler(enabled=enable_profiling) if enable_profiling else None

    # Generate data
    logger.info("\n" + "=" * 80)
    logger.info("DATA GENERATION")
    logger.info("=" * 80)

    train_data, eval_data = generate_synthetic_inventory_data(
        num_samples=num_samples,
        seq_len=seq_len,
        seed=seed
    )

    # Create model
    logger.info("\n" + "=" * 80)
    logger.info("MODEL CREATION")
    logger.info("=" * 80)

    if model_type == 'simple':
        logger.info("Creating Simple RNN model...")
        model = create_simple_model(seq_len=seq_len, output_size=1)
    elif model_type == 'transformer':
        logger.info("Creating Transformer model...")
        model = create_transformer_model(
            seq_len=seq_len,
            d_model=64,
            num_heads=4,
            num_layers=2,
            output_size=1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info(f"Model type: {model_type}")

    # Create optimizer
    optimizer = optax.adam(learning_rate)

    # Define loss function
    def mse_loss(pred, target):
        return jnp.mean((pred - target) ** 2)

    # Create distributed trainer
    logger.info("\n" + "=" * 80)
    logger.info("DISTRIBUTED TRAINER SETUP")
    logger.info("=" * 80)

    logger.info(f"Strategy: {strategy}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {num_epochs}")

    if profiler:
        with profiler.profile("trainer_initialization"):
            trainer = DistributedTrainer(
                model=model,
                optimizer=optimizer,
                loss_fn=mse_loss,
                strategy=strategy
            )
    else:
        trainer = DistributedTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=mse_loss,
            strategy=strategy
        )

    trainer.log_info()

    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)

    start_time = time.time()

    if profiler:
        with profiler.profile("training", epochs=num_epochs, batch_size=batch_size):
            state, history = trainer.train(
                train_data=train_data,
                num_epochs=num_epochs,
                batch_size=batch_size,
                input_shape=(1, seq_len, 1),
                eval_data=eval_data,
                log_every=max(1, num_epochs // 10),
                seed=seed
            )
    else:
        state, history = trainer.train(
            train_data=train_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            input_shape=(1, seq_len, 1),
            eval_data=eval_data,
            log_every=max(1, num_epochs // 10),
            seed=seed
        )

    training_time = time.time() - start_time

    # Results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)

    logger.info(f"Total training time: {training_time:.2f}s")
    logger.info(f"Time per epoch: {training_time/num_epochs:.2f}s")

    if history.get('train_loss'):
        logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"Initial train loss: {history['train_loss'][0]:.6f}")
        logger.info(f"Loss reduction: {(1 - history['train_loss'][-1]/history['train_loss'][0])*100:.2f}%")

    if history.get('eval_loss'):
        logger.info(f"Final eval loss: {history['eval_loss'][-1]:.6f}")
        logger.info(f"Best eval loss: {min(history['eval_loss']):.6f}")

    # Profiling results
    if profiler:
        logger.info("\n" + "=" * 80)
        logger.info("PROFILING RESULTS")
        logger.info("=" * 80)
        profiler.print_summary()

    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)

        results = {
            'experiment': {
                'strategy': strategy,
                'model_type': model_type,
                'num_samples': num_samples,
                'seq_len': seq_len,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'seed': seed,
                'timestamp': timestamp
            },
            'system': make_json_serializable(setup_distributed()),
            'trainer_info': make_json_serializable(trainer.get_trainer_info()),
            'results': {
                'training_time_s': training_time,
                'time_per_epoch_s': training_time / num_epochs,
                'history': {
                    k: [float(v) for v in vals] for k, vals in history.items()
                }
            }
        }

        if profiler:
            results['profiling'] = {
                'summary': make_json_serializable(profiler.get_summary()),
                'measurements': make_json_serializable(profiler.get_measurements())
            }

        results_file = results_dir / f"distributed_training_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")

        # Also save profiling data separately if available
        if profiler:
            profile_file = results_dir / f"profile_{timestamp}.json"
            profiler.save_to_file(profile_file)

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 80)

    return trainer, state, history


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Distributed Training Experiment for JAX Inventory Optimizer'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default='auto',
        choices=['auto', 'data', 'model', 'hybrid'],
        help='Parallelism strategy (default: auto)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='simple',
        choices=['simple', 'transformer'],
        help='Model type (default: simple)'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of training samples (default: 1000)'
    )

    parser.add_argument(
        '--seq-len',
        type=int,
        default=50,
        help='Sequence length (default: 50)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Global batch size (default: 32)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )

    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable GPU profiling'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    try:
        run_distributed_training_experiment(
            strategy=args.strategy,
            model_type=args.model,
            num_samples=args.samples,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            enable_profiling=args.profile,
            save_results=not args.no_save,
            seed=args.seed
        )
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
