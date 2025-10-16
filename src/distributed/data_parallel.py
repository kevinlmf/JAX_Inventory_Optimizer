"""
Data Parallel Training using JAX pmap

This module implements data parallelism where each device processes
a different batch of data with the same model.
"""

import jax
import jax.numpy as jnp
from jax import pmap, lax
import optax
from typing import Dict, Any, Callable, Optional, Tuple
from flax import linen as nn
from flax.training import train_state
import logging

logger = logging.getLogger(__name__)


class DataParallelTrainer:
    """
    Data parallel trainer using JAX pmap

    This trainer replicates the model on each device and processes
    different data batches in parallel.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        num_devices: Optional[int] = None
    ):
        """
        Initialize data parallel trainer

        Args:
            model: Flax neural network module
            optimizer: Optax optimizer
            loss_fn: Loss function
            num_devices: Number of devices (None = all available)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        if num_devices is None:
            self.num_devices = jax.device_count()
        else:
            self.num_devices = num_devices

        logger.info(f"DataParallelTrainer initialized with {self.num_devices} devices")

    def create_train_state(
        self,
        rng: jnp.ndarray,
        input_shape: Tuple[int, ...]
    ) -> train_state.TrainState:
        """
        Create initial training state

        Args:
            rng: Random key
            input_shape: Shape of input data

        Returns:
            Training state
        """
        # Initialize model parameters
        dummy_input = jnp.ones(input_shape)

        # Split RNG for initialization
        init_rng, dropout_rng = jax.random.split(rng)

        # Initialize with dropout RNG
        variables = self.model.init(
            {'params': init_rng, 'dropout': dropout_rng},
            dummy_input,
            training=False
        )

        # Create train state
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=self.optimizer
        )

        return state

    def replicate_state(self, state: train_state.TrainState) -> train_state.TrainState:
        """
        Replicate training state across devices

        Args:
            state: Training state

        Returns:
            Replicated state
        """
        return jax.device_put_replicated(state, jax.devices()[:self.num_devices])

    @staticmethod
    def create_train_step() -> Callable:
        """
        Create parallelized training step function

        Returns:
            pmap'd training step function
        """

        def train_step(state, batch, rng):
            """Single training step on one device"""

            def loss_fn_wrapper(params):
                # Pass RNG for dropout and training=True
                predictions = state.apply_fn(
                    {'params': params},
                    batch['input'],
                    training=True,
                    rngs={'dropout': rng}
                )
                loss = jnp.mean((predictions - batch['target']) ** 2)
                return loss, predictions

            # Compute gradients
            grad_fn = jax.value_and_grad(loss_fn_wrapper, has_aux=True)
            (loss, predictions), grads = grad_fn(state.params)

            # Synchronize gradients across devices
            grads = lax.pmean(grads, axis_name='data')

            # Update parameters
            state = state.apply_gradients(grads=grads)

            # Compute metrics
            metrics = {
                'loss': loss,
                'mse': jnp.mean((predictions - batch['target']) ** 2)
            }

            # Average metrics across devices
            metrics = lax.pmean(metrics, axis_name='data')

            return state, metrics

        # Parallelize across devices
        return pmap(train_step, axis_name='data')

    @staticmethod
    def create_eval_step() -> Callable:
        """
        Create parallelized evaluation step function

        Returns:
            pmap'd evaluation step function
        """

        def eval_step(state, batch):
            """Single evaluation step on one device"""
            # Use training=False for deterministic evaluation
            predictions = state.apply_fn(
                {'params': state.params},
                batch['input'],
                training=False
            )

            metrics = {
                'loss': jnp.mean((predictions - batch['target']) ** 2),
                'mae': jnp.mean(jnp.abs(predictions - batch['target']))
            }

            # Average metrics across devices
            metrics = lax.pmean(metrics, axis_name='data')

            return metrics

        return pmap(eval_step, axis_name='data')

    def train(
        self,
        state: train_state.TrainState,
        train_data: Dict[str, jnp.ndarray],
        num_epochs: int,
        eval_data: Optional[Dict[str, jnp.ndarray]] = None,
        log_every: int = 10,
        rng_seed: int = 42
    ) -> Tuple[train_state.TrainState, Dict[str, list]]:
        """
        Train model with data parallelism

        Args:
            state: Initial training state (already replicated)
            train_data: Training data (already sharded)
            num_epochs: Number of training epochs
            eval_data: Optional evaluation data (already sharded)
            log_every: Log frequency
            rng_seed: Random seed for dropout

        Returns:
            Tuple of (final_state, training_history)
        """
        train_step_fn = self.create_train_step()
        eval_step_fn = self.create_eval_step() if eval_data else None

        history = {'train_loss': [], 'eval_loss': []}

        num_batches = train_data['input'].shape[0]

        # Initialize RNG
        rng = jax.random.PRNGKey(rng_seed)

        for epoch in range(num_epochs):
            epoch_metrics = []

            # Training loop
            for batch_idx in range(num_batches):
                # Split RNG for this step
                rng, step_rng = jax.random.split(rng)

                # Replicate RNG across devices
                step_rng_replicated = jax.random.split(step_rng, self.num_devices)

                batch = {
                    'input': train_data['input'][batch_idx],
                    'target': train_data['target'][batch_idx]
                }

                state, metrics = train_step_fn(state, batch, step_rng_replicated)
                epoch_metrics.append(metrics)

            # Average metrics over epoch
            epoch_loss = jnp.mean(jnp.array([m['loss'][0] for m in epoch_metrics]))
            history['train_loss'].append(float(epoch_loss))

            # Evaluation
            if eval_data is not None and eval_step_fn is not None:
                eval_metrics = []
                num_eval_batches = eval_data['input'].shape[0]

                for batch_idx in range(num_eval_batches):
                    batch = {
                        'input': eval_data['input'][batch_idx],
                        'target': eval_data['target'][batch_idx]
                    }
                    metrics = eval_step_fn(state, batch)
                    eval_metrics.append(metrics)

                eval_loss = jnp.mean(jnp.array([m['loss'][0] for m in eval_metrics]))
                history['eval_loss'].append(float(eval_loss))

            # Logging
            if (epoch + 1) % log_every == 0:
                log_msg = f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f}"
                if eval_data is not None:
                    log_msg += f" - Eval Loss: {eval_loss:.4f}"
                logger.info(log_msg)

        return state, history

    def unreplicate_state(self, state: train_state.TrainState) -> train_state.TrainState:
        """
        Unreplicate state from devices (take first replica)

        Args:
            state: Replicated training state

        Returns:
            Unreplicated state
        """
        return jax.tree.map(lambda x: x[0], state)


class MultiSKUDataParallelTrainer(DataParallelTrainer):
    """
    Specialized data parallel trainer for multiple SKUs

    Each device trains on different SKUs in parallel
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        num_skus: int,
        num_devices: Optional[int] = None
    ):
        super().__init__(model, optimizer, loss_fn, num_devices)
        self.num_skus = num_skus

        # Calculate SKUs per device
        self.skus_per_device = num_skus // self.num_devices
        if num_skus % self.num_devices != 0:
            logger.warning(
                f"num_skus ({num_skus}) not divisible by num_devices ({self.num_devices}). "
                f"Will use {self.skus_per_device * self.num_devices} SKUs."
            )

    def prepare_multi_sku_data(
        self,
        data: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Prepare multi-SKU data for parallel processing

        Args:
            data: Dictionary with 'input' and 'target' arrays
                  Shape: [num_skus, num_timesteps, features]

        Returns:
            Sharded data ready for pmap
        """
        # Reshape to [num_devices, skus_per_device, num_timesteps, features]
        sharded_data = {}

        for key, value in data.items():
            # Trim to divisible size
            value = value[:self.skus_per_device * self.num_devices]

            # Reshape for parallel processing
            value = value.reshape(
                self.num_devices,
                self.skus_per_device,
                *value.shape[1:]
            )

            sharded_data[key] = value

        return sharded_data


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Simple model for testing
    class SimpleModel(nn.Module):
        features: int = 64

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.features)(x)
            x = nn.relu(x)
            x = nn.Dense(1)(x)
            return x

    # Test data parallel training
    model = SimpleModel()
    optimizer = optax.adam(1e-3)

    def mse_loss(predictions, targets):
        return jnp.mean((predictions - targets) ** 2)

    trainer = DataParallelTrainer(model, optimizer, mse_loss)

    logger.info(f"Trainer initialized with {trainer.num_devices} devices")
