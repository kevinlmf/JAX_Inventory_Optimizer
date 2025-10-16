"""
Model Parallel Training using JAX pjit

This module implements model parallelism where the model is sharded
across devices for training large models.
"""

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, PartitionSpec as P, NamedSharding
from jax.experimental.pjit import pjit
import optax
from typing import Dict, Any, Callable, Optional, Tuple
from flax import linen as nn
from flax.training import train_state
import logging

logger = logging.getLogger(__name__)


class ModelParallelTrainer:
    """
    Model parallel trainer using JAX pjit

    This trainer shards the model across devices for training models
    that don't fit in a single device's memory.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        mesh_shape: Tuple[int, ...] = None,
        num_devices: Optional[int] = None
    ):
        """
        Initialize model parallel trainer

        Args:
            model: Flax neural network module
            optimizer: Optax optimizer
            loss_fn: Loss function
            mesh_shape: Shape of device mesh (e.g., (2, 4) for 8 devices)
            num_devices: Number of devices (None = all available)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        if num_devices is None:
            self.num_devices = jax.device_count()
        else:
            self.num_devices = num_devices

        if mesh_shape is None:
            # Default: 2D mesh for data + model parallelism
            if self.num_devices >= 8:
                mesh_shape = (2, self.num_devices // 2)
            elif self.num_devices >= 4:
                mesh_shape = (2, self.num_devices // 2)
            else:
                mesh_shape = (self.num_devices,)

        self.mesh_shape = mesh_shape
        self.mesh = self._create_mesh()

        logger.info(
            f"ModelParallelTrainer initialized with {self.num_devices} devices "
            f"in mesh shape {mesh_shape}"
        )

    def _create_mesh(self) -> Mesh:
        """Create device mesh for model parallelism"""
        devices = mesh_utils.create_device_mesh(
            self.mesh_shape,
            jax.devices()[:self.num_devices]
        )

        # Axis names based on mesh dimensionality
        if len(self.mesh_shape) == 1:
            axis_names = ('model',)
        elif len(self.mesh_shape) == 2:
            axis_names = ('data', 'model')
        else:
            axis_names = tuple(f'axis_{i}' for i in range(len(self.mesh_shape)))

        return Mesh(devices, axis_names)

    def create_sharding_spec(
        self,
        data_axis: Optional[str] = None,
        model_axis: Optional[str] = None
    ) -> Dict[str, PartitionSpec]:
        """
        Create sharding specification for parameters and data

        Args:
            data_axis: Name of data parallelism axis
            model_axis: Name of model parallelism axis

        Returns:
            Dictionary of partition specs
        """
        # Default sharding strategy
        return {
            'params': P(model_axis),  # Shard parameters along model axis
            'data': P(data_axis, None),  # Shard data along batch axis
            'optimizer': P(model_axis),  # Shard optimizer state
        }

    def create_train_state(
        self,
        rng: jnp.ndarray,
        input_shape: Tuple[int, ...],
        sharding_spec: Optional[Dict] = None
    ) -> train_state.TrainState:
        """
        Create initial training state with sharding

        Args:
            rng: Random key
            input_shape: Shape of input data
            sharding_spec: Custom sharding specification

        Returns:
            Training state with sharding
        """
        if sharding_spec is None:
            sharding_spec = self.create_sharding_spec()

        # Initialize model parameters
        dummy_input = jnp.ones(input_shape)

        with self.mesh:
            variables = self.model.init(rng, dummy_input)

            # Create train state
            state = train_state.TrainState.create(
                apply_fn=self.model.apply,
                params=variables['params'],
                tx=self.optimizer
            )

        return state

    def create_train_step(
        self,
        in_shardings: Tuple[PartitionSpec, ...],
        out_shardings: Tuple[PartitionSpec, ...]
    ) -> Callable:
        """
        Create sharded training step function

        Args:
            in_shardings: Input sharding specs (state, batch)
            out_shardings: Output sharding specs (state, metrics)

        Returns:
            pjit'd training step function
        """

        @pjit(
            in_shardings=in_shardings,
            out_shardings=out_shardings
        )
        def train_step(state, batch):
            """Single training step with model parallelism"""

            def loss_fn_wrapper(params):
                predictions = state.apply_fn({'params': params}, batch['input'])
                loss = jnp.mean((predictions - batch['target']) ** 2)
                return loss, predictions

            # Compute gradients
            grad_fn = jax.value_and_grad(loss_fn_wrapper, has_aux=True)
            (loss, predictions), grads = grad_fn(state.params)

            # Update parameters
            state = state.apply_gradients(grads=grads)

            # Compute metrics
            metrics = {
                'loss': loss,
                'mse': jnp.mean((predictions - batch['target']) ** 2)
            }

            return state, metrics

        return train_step

    def train(
        self,
        state: train_state.TrainState,
        train_data: Dict[str, jnp.ndarray],
        num_epochs: int,
        batch_size: int,
        eval_data: Optional[Dict[str, jnp.ndarray]] = None,
        log_every: int = 10
    ) -> Tuple[train_state.TrainState, Dict[str, list]]:
        """
        Train model with model parallelism

        Args:
            state: Initial training state
            train_data: Training data
            num_epochs: Number of training epochs
            batch_size: Batch size
            eval_data: Optional evaluation data
            log_every: Log frequency

        Returns:
            Tuple of (final_state, training_history)
        """
        # Create sharding specs
        state_sharding = NamedSharding(self.mesh, P('model'))
        data_sharding = NamedSharding(self.mesh, P('data', None))
        metrics_sharding = NamedSharding(self.mesh, P())

        train_step_fn = self.create_train_step(
            in_shardings=(state_sharding, data_sharding),
            out_shardings=(state_sharding, metrics_sharding)
        )

        history = {'train_loss': []}

        num_samples = train_data['input'].shape[0]
        num_batches = num_samples // batch_size

        with self.mesh:
            for epoch in range(num_epochs):
                epoch_metrics = []

                # Training loop
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size

                    batch = {
                        'input': train_data['input'][start_idx:end_idx],
                        'target': train_data['target'][start_idx:end_idx]
                    }

                    state, metrics = train_step_fn(state, batch)
                    epoch_metrics.append(metrics)

                # Average metrics over epoch
                epoch_loss = jnp.mean(jnp.array([m['loss'] for m in epoch_metrics]))
                history['train_loss'].append(float(epoch_loss))

                # Logging
                if (epoch + 1) % log_every == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f}"
                    )

        return state, history


class HybridParallelTrainer:
    """
    Hybrid parallel trainer combining data and model parallelism

    Uses 2D mesh: one dimension for data parallelism, one for model parallelism
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        data_parallel_size: int = 2,
        model_parallel_size: int = 4
    ):
        """
        Initialize hybrid parallel trainer

        Args:
            model: Flax neural network module
            optimizer: Optax optimizer
            loss_fn: Loss function
            data_parallel_size: Number of devices for data parallelism
            model_parallel_size: Number of devices for model parallelism
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.data_parallel_size = data_parallel_size
        self.model_parallel_size = model_parallel_size
        self.total_devices = data_parallel_size * model_parallel_size

        assert self.total_devices <= jax.device_count(), \
            f"Requested {self.total_devices} devices but only {jax.device_count()} available"

        self.mesh = self._create_2d_mesh()

        logger.info(
            f"HybridParallelTrainer initialized: "
            f"{data_parallel_size} data parallel x {model_parallel_size} model parallel"
        )

    def _create_2d_mesh(self) -> Mesh:
        """Create 2D device mesh for hybrid parallelism"""
        mesh_shape = (self.data_parallel_size, self.model_parallel_size)
        devices = mesh_utils.create_device_mesh(
            mesh_shape,
            jax.devices()[:self.total_devices]
        )

        return Mesh(devices, ('data', 'model'))

    def create_sharding_strategy(self) -> Dict[str, PartitionSpec]:
        """
        Create sharding strategy for hybrid parallelism

        Returns:
            Dictionary mapping component names to partition specs
        """
        return {
            # Replicate across data axis, shard across model axis
            'params': P(None, 'model'),

            # Shard batch across data axis, features across model axis
            'activations': P('data', 'model'),

            # Replicate batch across data axis, shard weights across model
            'weights': P(None, 'model'),

            # Shard optimizer state same as params
            'opt_state': P(None, 'model'),
        }

    def train_step_hybrid(self, state, batch):
        """
        Training step with hybrid parallelism

        This combines:
        - Data parallelism: Each replica processes different data
        - Model parallelism: Model weights are sharded across devices
        """

        def loss_fn_wrapper(params):
            predictions = state.apply_fn({'params': params}, batch['input'])
            loss = jnp.mean((predictions - batch['target']) ** 2)
            return loss, predictions

        # Compute gradients (automatically handles sharding)
        grad_fn = jax.value_and_grad(loss_fn_wrapper, has_aux=True)
        (loss, predictions), grads = grad_fn(state.params)

        # Synchronize gradients across data parallel replicas
        grads = jax.lax.pmean(grads, axis_name='data')

        # Update parameters
        state = state.apply_gradients(grads=grads)

        metrics = {
            'loss': jax.lax.pmean(loss, axis_name='data'),
            'mse': jax.lax.pmean(
                jnp.mean((predictions - batch['target']) ** 2),
                axis_name='data'
            )
        }

        return state, metrics


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Simple model for testing
    class LargeModel(nn.Module):
        features: Tuple[int, ...] = (1024, 2048, 1024)

        @nn.compact
        def __call__(self, x):
            for feat in self.features:
                x = nn.Dense(feat)(x)
                x = nn.relu(x)
            x = nn.Dense(1)(x)
            return x

    # Test model parallel training
    model = LargeModel()
    optimizer = optax.adam(1e-3)

    def mse_loss(predictions, targets):
        return jnp.mean((predictions - targets) ** 2)

    trainer = ModelParallelTrainer(model, optimizer, mse_loss)

    logger.info(f"Trainer initialized with mesh shape {trainer.mesh_shape}")
