"""
Unified distributed trainer interface

This module provides a high-level interface for distributed training
that automatically selects the appropriate parallelism strategy.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple, Callable, Literal, NamedTuple
from flax import linen as nn
import optax
import logging

from .data_parallel import DataParallelTrainer
from .model_parallel import ModelParallelTrainer, HybridParallelTrainer
from .utils import setup_distributed, log_distributed_info

logger = logging.getLogger(__name__)


class TrainState(NamedTuple):
    """Simple training state as a NamedTuple (JAX pytree compatible)"""
    params: Any
    opt_state: Any
    step: int


class DistributedTrainer:
    """
    Unified interface for distributed training

    Automatically selects the best parallelism strategy based on:
    - Model size
    - Number of available devices
    - User preferences
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        strategy: Literal['auto', 'data', 'model', 'hybrid'] = 'auto',
        num_devices: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize distributed trainer

        Args:
            model: Flax neural network module
            optimizer: Optax optimizer
            loss_fn: Loss function
            strategy: Parallelism strategy:
                - 'auto': Automatically select best strategy
                - 'data': Data parallelism only
                - 'model': Model parallelism only
                - 'hybrid': Hybrid data + model parallelism
            num_devices: Number of devices (None = all available)
            **kwargs: Additional arguments for specific trainers
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # Setup distributed environment
        self.dist_info = setup_distributed()
        if num_devices is None:
            self.num_devices = self.dist_info['device_count']
        else:
            self.num_devices = min(num_devices, self.dist_info['device_count'])

        # Select strategy
        if strategy == 'auto':
            strategy = self._auto_select_strategy(**kwargs)

        self.strategy = strategy
        self.trainer = self._create_trainer(**kwargs)

        logger.info(f"DistributedTrainer initialized with strategy: {strategy}")

    def _auto_select_strategy(self, **kwargs) -> str:
        """
        Automatically select the best parallelism strategy

        Heuristics:
        - Single device: No parallelism needed
        - 2-4 devices: Data parallelism
        - 8+ devices: Hybrid parallelism
        - Large model: Model or hybrid parallelism
        """
        if self.num_devices == 1:
            logger.info("Single device detected, using data parallelism")
            return 'data'

        # Try to estimate model size
        try:
            model_params = self._estimate_model_size()
            logger.info(f"Estimated model parameters: {model_params:,}")

            # Heuristic: > 100M parameters suggests model parallelism
            if model_params > 100_000_000:
                if self.num_devices >= 8:
                    logger.info("Large model + many devices -> hybrid parallelism")
                    return 'hybrid'
                else:
                    logger.info("Large model -> model parallelism")
                    return 'model'
        except Exception as e:
            logger.warning(f"Could not estimate model size: {e}")

        # Default strategies based on device count
        if self.num_devices >= 8:
            logger.info("8+ devices -> hybrid parallelism")
            return 'hybrid'
        elif self.num_devices >= 4:
            logger.info("4+ devices -> data parallelism")
            return 'data'
        else:
            logger.info("2-3 devices -> data parallelism")
            return 'data'

    def _estimate_model_size(self) -> int:
        """
        Estimate number of parameters in model

        Returns:
            Estimated parameter count
        """
        # Create dummy input to initialize model
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 10))  # Dummy shape

        try:
            variables = self.model.init(rng, dummy_input)
            params = variables['params']

            # Count parameters
            param_count = sum(
                x.size for x in jax.tree_util.tree_leaves(params)
            )
            return param_count
        except Exception as e:
            logger.warning(f"Could not initialize model for size estimation: {e}")
            return 0

    def _create_trainer(self, **kwargs):
        """Create specific trainer based on strategy"""
        if self.strategy == 'data':
            return DataParallelTrainer(
                self.model,
                self.optimizer,
                self.loss_fn,
                num_devices=self.num_devices,
                **kwargs
            )
        elif self.strategy == 'model':
            return ModelParallelTrainer(
                self.model,
                self.optimizer,
                self.loss_fn,
                num_devices=self.num_devices,
                **kwargs
            )
        elif self.strategy == 'hybrid':
            # Extract hybrid-specific args
            data_parallel_size = kwargs.get('data_parallel_size', 2)
            model_parallel_size = self.num_devices // data_parallel_size

            return HybridParallelTrainer(
                self.model,
                self.optimizer,
                self.loss_fn,
                data_parallel_size=data_parallel_size,
                model_parallel_size=model_parallel_size
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def train(
        self,
        train_data: Dict[str, jnp.ndarray],
        num_epochs: int,
        batch_size: int,
        input_shape: Tuple[int, ...],
        eval_data: Optional[Dict[str, jnp.ndarray]] = None,
        log_every: int = 10,
        seed: int = 42
    ) -> Tuple[Any, Dict[str, list]]:
        """
        Train model with distributed strategy

        Args:
            train_data: Training data dictionary
            num_epochs: Number of epochs
            batch_size: Batch size (global)
            input_shape: Shape of model input
            eval_data: Optional evaluation data
            log_every: Logging frequency
            seed: Random seed

        Returns:
            Tuple of (trained_state, history)
        """
        rng = jax.random.PRNGKey(seed)

        # Create initial state
        state = self.trainer.create_train_state(rng, input_shape)

        # Different strategies have different training interfaces
        if self.strategy == 'data':
            # Data parallelism requires replicated state and sharded data
            state = self.trainer.replicate_state(state)

            from .utils import create_data_loader_parallel

            # Prepare sharded data
            train_data_sharded, num_batches = create_data_loader_parallel(
                train_data['input'],
                batch_size,
                self.num_devices,
                shuffle=True,
                seed=seed
            )

            train_data_dict = {
                'input': train_data_sharded,
                'target': create_data_loader_parallel(
                    train_data['target'],
                    batch_size,
                    self.num_devices,
                    shuffle=True,
                    seed=seed
                )[0]
            }

            # Prepare sharded eval data if provided
            eval_data_dict = None
            if eval_data is not None:
                eval_data_sharded, num_eval_batches = create_data_loader_parallel(
                    eval_data['input'],
                    batch_size,
                    self.num_devices,
                    shuffle=False,
                    seed=seed
                )

                eval_data_dict = {
                    'input': eval_data_sharded,
                    'target': create_data_loader_parallel(
                        eval_data['target'],
                        batch_size,
                        self.num_devices,
                        shuffle=False,
                        seed=seed
                    )[0]
                }

            # Train
            state, history = self.trainer.train(
                state,
                train_data_dict,
                num_epochs,
                eval_data=eval_data_dict,
                log_every=log_every
            )

            # Unreplicate state
            state = self.trainer.unreplicate_state(state)

        elif self.strategy in ['model', 'hybrid']:
            # Model/hybrid parallelism
            state, history = self.trainer.train(
                state,
                train_data,
                num_epochs,
                batch_size,
                eval_data=eval_data,
                log_every=log_every
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return state, history

    def get_trainer_info(self) -> Dict[str, Any]:
        """
        Get information about the distributed trainer

        Returns:
            Dictionary with trainer configuration
        """
        return {
            'strategy': self.strategy,
            'num_devices': self.num_devices,
            'device_type': self.dist_info['platform'],
            'model': str(self.model),
            'optimizer': str(self.optimizer),
        }

    def log_info(self):
        """Log trainer information"""
        log_distributed_info()

        info = self.get_trainer_info()
        logger.info("=" * 60)
        logger.info("Trainer Configuration")
        logger.info("=" * 60)
        for key, value in info.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 60)


class DistributedRLTrainer(DistributedTrainer):
    """
    Specialized distributed trainer for reinforcement learning

    Handles RL-specific requirements like:
    - Multiple parallel environments
    - Experience replay buffer distribution
    - Asynchronous policy updates
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        num_envs: int,
        strategy: Literal['auto', 'data', 'model', 'hybrid'] = 'auto',
        **kwargs
    ):
        """
        Initialize distributed RL trainer

        Args:
            model: RL policy/value network
            optimizer: Optax optimizer
            loss_fn: RL loss function
            num_envs: Number of parallel environments
            strategy: Parallelism strategy
        """
        super().__init__(model, optimizer, loss_fn, strategy, **kwargs)
        self.num_envs = num_envs

        # Calculate envs per device
        self.envs_per_device = num_envs // self.num_devices
        if num_envs % self.num_devices != 0:
            logger.warning(
                f"num_envs ({num_envs}) not divisible by num_devices ({self.num_devices})"
            )

    def train_rl(
        self,
        env_fn: Callable,
        num_steps: int,
        input_shape: Tuple[int, ...],
        **kwargs
    ):
        """
        Train RL agent with distributed environments

        Args:
            env_fn: Function to create environment instances
            num_steps: Total training steps
            input_shape: Shape of observations
            **kwargs: Additional training arguments
        """
        # Implementation for distributed RL training
        # This would involve:
        # 1. Creating parallel environments on each device
        # 2. Collecting experiences in parallel
        # 3. Synchronizing policy updates
        logger.info(f"Training RL agent with {self.num_envs} parallel environments")

        # TODO: Implement distributed RL training loop
        raise NotImplementedError("Distributed RL training coming soon")


def create_distributed_trainer(
    loss_fn: Callable,
    strategy: Literal['auto', 'data', 'model', 'hybrid'] = 'auto',
    learning_rate: float = 1e-3,
    **kwargs
):
    """
    Simplified factory function for creating a distributed trainer.

    This is a lightweight wrapper for use cases where you don't need
    the full DistributedTrainer class. It returns an object with
    train_step and eval_step methods.

    Args:
        loss_fn: Loss function that takes (params, batch) and returns loss
        strategy: Parallelism strategy ('auto', 'data', 'model', 'hybrid')
        learning_rate: Learning rate for optimizer
        **kwargs: Additional arguments

    Returns:
        SimpleDistributedTrainer with train_step, eval_step, create_train_state methods
    """
    return SimpleDistributedTrainer(loss_fn, strategy, learning_rate, **kwargs)


class SimpleDistributedTrainer:
    """
    Simplified distributed trainer for functional-style training.

    This class provides a simpler interface than DistributedTrainer
    for cases where you're working with pure functions rather than
    Flax modules.
    """

    def __init__(
        self,
        loss_fn: Callable,
        strategy: Literal['auto', 'data', 'model', 'hybrid'] = 'auto',
        learning_rate: float = 1e-3,
        **kwargs
    ):
        self.loss_fn = loss_fn
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.optimizer = optax.adam(learning_rate)

        # Setup distributed environment
        from .utils import setup_distributed
        self.dist_info = setup_distributed()
        self.num_devices = self.dist_info['device_count']

        # Will create jitted functions lazily when needed
        self._train_step_fn = None
        self._eval_step_fn = None

        logger.info(f"SimpleDistributedTrainer initialized with strategy: {strategy}")
        logger.info(f"Available devices: {self.num_devices}")

    def create_train_state(self, params, learning_rate: Optional[float] = None):
        """
        Create a simple training state.

        Args:
            params: Model parameters
            learning_rate: Optional learning rate override

        Returns:
            TrainState with params, optimizer state, and step
        """
        if learning_rate is not None:
            self.optimizer = optax.adam(learning_rate)
            self.learning_rate = learning_rate

        # Create jitted functions now that optimizer is finalized
        if self._train_step_fn is None:
            self._train_step_fn = self._make_train_step()
            self._eval_step_fn = self._make_eval_step()

        opt_state = self.optimizer.init(params)

        # Return a TrainState dataclass
        return TrainState(
            params=params,
            opt_state=opt_state,
            step=0
        )

    def _make_train_step(self):
        """Create jitted train step function"""
        loss_fn = self.loss_fn
        optimizer = self.optimizer

        @jax.jit
        def train_step(state, batch):
            """Single device training step"""
            params = state.params

            def loss_wrapper(p):
                return loss_fn(p, batch)

            loss, grads = jax.value_and_grad(loss_wrapper)(params)

            # Apply gradients
            updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
            new_params = optax.apply_updates(params, updates)

            new_state = TrainState(
                params=new_params,
                opt_state=new_opt_state,
                step=state.step + 1
            )

            metrics = {'loss': loss}
            return new_state, metrics

        return train_step

    def _make_eval_step(self):
        """Create jitted eval step function"""
        loss_fn = self.loss_fn

        @jax.jit
        def eval_step(state, batch):
            """Single device evaluation step"""
            params = state.params
            loss = loss_fn(params, batch)
            return {'loss': loss}

        return eval_step

    def train_step(self, state, batch):
        """
        Perform a training step.

        Args:
            state: Training state
            batch: Batch of data (features, targets)

        Returns:
            Tuple of (new_state, metrics)
        """
        if self.num_devices > 1 and self.strategy == 'data':
            # Data parallel training
            # Replicate state and shard batch across devices
            # For now, fall back to single device
            logger.warning("Multi-device data parallelism not fully implemented, using single device")

        return self._train_step_fn(state, batch)

    def eval_step(self, state, batch):
        """
        Perform an evaluation step.

        Args:
            state: Training state
            batch: Batch of data (features, targets)

        Returns:
            Dictionary of metrics
        """
        return self._eval_step_fn(state, batch)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Simple model
    class TestModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(64)(x)
            x = nn.relu(x)
            x = nn.Dense(1)(x)
            return x

    model = TestModel()
    optimizer = optax.adam(1e-3)

    def mse_loss(pred, target):
        return jnp.mean((pred - target) ** 2)

    # Create trainer with auto strategy selection
    trainer = DistributedTrainer(
        model,
        optimizer,
        mse_loss,
        strategy='auto'
    )

    trainer.log_info()
