"""
Configuration management for the inventory optimization system
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class InventoryConfig:
    """Configuration for inventory problem parameters"""
    holding_cost: float = 2.0
    stockout_cost: float = 10.0
    ordering_cost: float = 50.0
    lead_time: int = 7
    service_level: float = 0.95
    max_inventory: int = 1000


@dataclass
class TraditionalConfig:
    """Configuration for traditional methods"""
    eoq: Dict[str, Any] = field(default_factory=lambda: {
        'holding_cost': 2.0,
        'ordering_cost': 50.0,
        'service_level': 0.95
    })

    safety_stock: Dict[str, Any] = field(default_factory=lambda: {
        'service_level': 0.95,
        'lead_time': 7,
        'review_period': 1
    })

    s_S_policy: Dict[str, Any] = field(default_factory=lambda: {
        'reorder_point': 50,
        'order_up_to': 200,
        'lead_time': 7
    })


@dataclass
class MLConfig:
    """Configuration for ML methods"""
    lstm: Dict[str, Any] = field(default_factory=lambda: {
        'sequence_length': 30,
        'hidden_size': 64,
        'num_layers': 2,
        'learning_rate': 0.001,
        'epochs': 100
    })

    transformer: Dict[str, Any] = field(default_factory=lambda: {
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'learning_rate': 0.0001,
        'epochs': 150
    })

    xgboost: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8
    })


@dataclass
class RLConfig:
    """Configuration for RL methods"""
    dqn: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_size': 256,
        'learning_rate': 0.001,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 10000,
        'batch_size': 32,
        'target_update': 10
    })

    ppo: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_size': 256,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01
    })

    sac: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_size': 256,
        'learning_rate': 3e-4,
        'alpha': 0.2,
        'gamma': 0.99,
        'tau': 0.005
    })


@dataclass
class ExperimentConfig:
    """Configuration for experiments and evaluation"""
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 42
    num_runs: int = 5

    scenarios: Dict[str, Any] = field(default_factory=lambda: {
        'base': {'demand_volatility': 0.2, 'seasonality': 0.3},
        'high_volatility': {'demand_volatility': 0.5, 'seasonality': 0.3},
        'seasonal': {'demand_volatility': 0.2, 'seasonality': 0.6},
        'promotional': {'demand_volatility': 0.2, 'seasonality': 0.3, 'promotion_freq': 0.1}
    })


class ConfigManager:
    """Manage configuration loading and saving"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            print(f"Config file {config_path} not found. Creating default config.")
            self.create_default_config(config_name)

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def save_config(self, config: Dict[str, Any], config_name: str) -> None:
        """Save configuration to YAML file"""
        config_path = self.config_dir / f"{config_name}.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        print(f"Config saved to {config_path}")

    def create_default_config(self, config_name: str) -> None:
        """Create default configuration file"""

        if config_name == "inventory":
            config = InventoryConfig().__dict__
        elif config_name == "traditional":
            config = TraditionalConfig().__dict__
        elif config_name == "ml":
            config = MLConfig().__dict__
        elif config_name == "rl":
            config = RLConfig().__dict__
        elif config_name == "experiment":
            config = ExperimentConfig().__dict__
        else:
            config = {}

        self.save_config(config, config_name)

    def get_method_config(self, method_category: str, method_name: str) -> Dict[str, Any]:
        """Get configuration for specific method"""

        config = self.load_config(method_category)
        return config.get(method_name, {})

    def update_method_config(self,
                           method_category: str,
                           method_name: str,
                           new_params: Dict[str, Any]) -> None:
        """Update configuration for specific method"""

        config = self.load_config(method_category)

        if method_name not in config:
            config[method_name] = {}

        config[method_name].update(new_params)
        self.save_config(config, method_category)


# Global config manager instance
config_manager = ConfigManager()


# Helper functions for easy access
def get_inventory_config() -> Dict[str, Any]:
    """Get inventory problem configuration"""
    return config_manager.load_config("inventory")


def get_traditional_config() -> Dict[str, Any]:
    """Get traditional methods configuration"""
    return config_manager.load_config("traditional")


def get_ml_config() -> Dict[str, Any]:
    """Get ML methods configuration"""
    return config_manager.load_config("ml")


def get_rl_config() -> Dict[str, Any]:
    """Get RL methods configuration"""
    return config_manager.load_config("rl")


def get_experiment_config() -> Dict[str, Any]:
    """Get experiment configuration"""
    return config_manager.load_config("experiment")


if __name__ == "__main__":
    # Create all default configs
    config_types = ["inventory", "traditional", "ml", "rl", "experiment"]

    print("Creating default configuration files...")

    for config_type in config_types:
        config_manager.create_default_config(config_type)

    print(f"\n✅ Created {len(config_types)} configuration files in 'configs/' directory")
    print("\nGenerated configs:")
    for config_type in config_types:
        print(f"  - configs/{config_type}.yaml")

    print("\nYou can now customize these configurations for your experiments!")