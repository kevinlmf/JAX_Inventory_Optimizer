# JAX Inventory Optimizer - System Architecture

## 🏗️ Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    JAX Inventory Optimizer                      │
├─────────────────────────────────────────────────────────────────┤
│                        API Layer                                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   FastAPI       │ │  Real-time      │ │   Batch         │   │
│  │   REST API      │ │  Recommendations │ │   Processing    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                   Comparison Framework                          │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  Performance    │ │   Benchmark     │ │   Visualization │   │
│  │   Evaluator     │ │    Runner       │ │     Dashboard   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Method Modules                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Traditional   │ │      ML/DL      │ │  Reinforcement  │   │
│  │    Methods      │ │     Methods     │ │   Learning      │   │
│  │                 │ │                 │ │                 │   │
│  │ • EOQ           │ │ • LSTM          │ │ • DQN           │   │
│  │ • Safety Stock  │ │ • Transformer   │ │ • PPO           │   │
│  │ • (s,S) Policy  │ │ • XGBoost       │ │ • SAC           │   │
│  │ • ARIMA         │ │ • Prophet       │ │ • Multi-Agent   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Core Engine                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Environment   │ │   JAX Compute   │ │   State         │   │
│  │   Simulator     │ │     Engine      │ │   Management    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      Data Layer                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Data Loader   │ │  Preprocessor   │ │    Feature      │   │
│  │                 │ │                 │ │   Engineering   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

```
JAX-Inventory-Optimizer/
├── src/
│   ├── core/                    # Core engine components
│   │   ├── environment.py       # Inventory simulation environment
│   │   ├── state_manager.py     # State management utilities
│   │   └── jax_engine.py        # JAX computation utilities
│   │
│   ├── methods/                 # Algorithm implementations
│   │   ├── traditional/         # Classical inventory methods
│   │   │   ├── eoq.py          # Economic Order Quantity
│   │   │   ├── safety_stock.py # Safety stock models
│   │   │   ├── s_S_policy.py   # (s,S) reorder policies
│   │   │   └── arima.py        # ARIMA forecasting
│   │   │
│   │   ├── ml_methods/         # Machine Learning approaches
│   │   │   ├── lstm.py         # LSTM demand forecasting
│   │   │   ├── transformer.py  # Transformer models
│   │   │   ├── xgboost.py      # Gradient boosting
│   │   │   └── prophet.py      # Facebook Prophet
│   │   │
│   │   └── rl_methods/         # Reinforcement Learning
│   │       ├── dqn.py          # Deep Q-Networks
│   │       ├── ppo.py          # Proximal Policy Optimization
│   │       ├── sac.py          # Soft Actor-Critic
│   │       └── multi_agent.py  # Multi-agent systems
│   │
│   ├── comparison/             # Comparison framework
│   │   ├── evaluator.py        # Performance evaluation
│   │   ├── benchmark.py        # Benchmarking utilities
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── visualizer.py       # Results visualization
│   │
│   ├── data/                   # Data processing
│   │   ├── loader.py           # Dataset loading
│   │   ├── preprocessor.py     # Data preprocessing
│   │   └── features.py         # Feature engineering
│   │
│   ├── api/                    # API service
│   │   ├── main.py             # FastAPI main application
│   │   ├── models.py           # Pydantic data models
│   │   └── routes/             # API route handlers
│   │       ├── traditional.py
│   │       ├── ml.py
│   │       └── rl.py
│   │
│   └── utils/                  # Shared utilities
│       ├── config.py           # Configuration management
│       ├── logging.py          # Logging utilities
│       └── validation.py       # Input validation
│
├── experiments/                # Experimental scripts
│   ├── compare_all_methods.py  # Full comparison experiment
│   ├── parameter_tuning.py     # Hyperparameter optimization
│   ├── scenario_analysis.py    # Different demand scenarios
│   └── ablation_studies.py     # Component analysis
│
├── notebooks/                  # Jupyter analysis notebooks
│   ├── data_exploration.ipynb  # Dataset analysis
│   ├── method_comparison.ipynb # Results comparison
│   └── visualization.ipynb     # Advanced visualizations
│
├── tests/                      # Unit and integration tests
│   ├── test_traditional.py
│   ├── test_ml_methods.py
│   ├── test_rl_methods.py
│   └── test_comparison.py
│
├── data/                       # Datasets
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Preprocessed data
│   └── synthetic/              # Generated data
│
├── results/                    # Experimental results
│   ├── benchmarks/             # Benchmark results
│   ├── comparisons/            # Method comparisons
│   └── visualizations/         # Generated plots
│
├── configs/                    # Configuration files
│   ├── traditional.yaml        # Traditional method configs
│   ├── ml.yaml                 # ML method configs
│   └── rl.yaml                 # RL method configs
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── ARCHITECTURE.md             # This file
└── setup.py                    # Package installation
```

## 🔧 Core Interfaces

### Base Inventory Method Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import jax.numpy as jnp

class InventoryMethod(ABC):
    """Base interface for all inventory optimization methods"""

    @abstractmethod
    def fit(self, demand_history: jnp.ndarray, **kwargs) -> None:
        """Train/fit the method on historical data"""
        pass

    @abstractmethod
    def recommend_action(self, current_state: Dict[str, Any]) -> float:
        """Recommend inventory action given current state"""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters/hyperparameters"""
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return method name for identification"""
        pass
```

### Evaluation Interface

```python
class InventoryEvaluator:
    """Unified evaluation framework for all methods"""

    def evaluate_method(
        self,
        method: InventoryMethod,
        test_data: Dict[str, jnp.ndarray],
        scenario_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate method performance"""
        return {
            'total_cost': float,
            'holding_cost': float,
            'stockout_cost': float,
            'ordering_cost': float,
            'service_level': float,
            'inventory_turnover': float,
            'forecast_accuracy': float
        }
```

## 📊 Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw Data  │───▶│ Preprocessor│───▶│  Features   │
│   (CSV)     │    │             │    │  (JAX)      │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
┌─────────────┐    ┌─────────────┐    ┌─────▼─────┐
│  Results    │◀───│   Methods   │◀───│Train/Test │
│ (Metrics)   │    │(Trad/ML/RL) │    │   Split   │
└─────────────┘    └─────────────┘    └───────────┘
       │
┌──────▼──────┐    ┌─────────────┐
│ Comparison  │───▶│   Reports   │
│ Framework   │    │& Visualize  │
└─────────────┘    └─────────────┘
```

## 🎯 Method Categories

### 1. Traditional Methods (`src/methods/traditional/`)

**Economic Order Quantity (EOQ)**
```python
class EOQMethod(InventoryMethod):
    def __init__(self, holding_cost: float, ordering_cost: float):
        self.h = holding_cost  # Annual holding cost per unit
        self.K = ordering_cost # Fixed cost per order

    def recommend_action(self, state):
        D = self.estimate_demand_rate(state['demand_history'])
        optimal_q = jnp.sqrt(2 * D * self.K / self.h)
        return optimal_q if state['inventory_level'] <= self.reorder_point else 0.0
```

**Safety Stock Models**
```python
class SafetyStockMethod(InventoryMethod):
    def __init__(self, service_level: float = 0.95):
        self.service_level = service_level
        self.safety_factor = scipy.stats.norm.ppf(service_level)
```

### 2. ML/DL Methods (`src/methods/ml_methods/`)

**LSTM Forecasting**
```python
class LSTMInventoryMethod(InventoryMethod):
    def __init__(self, sequence_length: int = 30, hidden_size: int = 64):
        self.model = self._build_lstm_model()
        self.demand_predictor = None

    def recommend_action(self, state):
        predicted_demand = self.predict_demand(state['demand_history'])
        return self.calculate_optimal_order(predicted_demand)
```

**Transformer-based Forecasting**
```python
class TransformerInventoryMethod(InventoryMethod):
    def __init__(self, d_model: int = 128, n_heads: int = 8):
        self.attention_model = self._build_transformer()
```

### 3. RL Methods (`src/methods/rl_methods/`)

**Deep Q-Network (DQN)**
```python
class DQNInventoryMethod(InventoryMethod):
    def __init__(self, state_dim: int, action_dim: int):
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()

    def recommend_action(self, state):
        q_values = self.q_network.apply(self.params, state)
        return jnp.argmax(q_values)
```

## 🔄 Extension Points

### Adding New Methods
1. Inherit from `InventoryMethod` base class
2. Implement required abstract methods
3. Add to appropriate category directory
4. Register in `comparison/benchmark.py`

### New Evaluation Metrics
1. Add to `comparison/metrics.py`
2. Update `evaluator.py` to compute new metrics
3. Modify visualization in `visualizer.py`

### Custom Demand Scenarios
1. Extend `core/environment.py`
2. Add scenario configs in `configs/`
3. Update `experiments/scenario_analysis.py`

## 🚀 Execution Workflow

### Single Method Execution
```python
# Load and preprocess data
data = DataLoader().load_dataset()
processed_data = Preprocessor().transform(data)

# Initialize method
method = EOQMethod(holding_cost=2.0, ordering_cost=50.0)
method.fit(processed_data['train'])

# Evaluate
evaluator = InventoryEvaluator()
results = evaluator.evaluate_method(method, processed_data['test'])
```

### Full Comparison
```python
# Run complete benchmark
from experiments.compare_all_methods import run_full_comparison

results = run_full_comparison(
    dataset='sample_retail_data',
    methods=['EOQ', 'LSTM', 'DQN', 'PPO'],
    scenarios=['base', 'high_volatility', 'seasonal']
)
```

## 🎨 Visualization & Reporting

- **Performance Dashboard**: Interactive Plotly dashboards
- **Method Comparison**: Side-by-side performance metrics
- **Time Series Analysis**: Demand vs. inventory level plots
- **Cost Breakdown**: Detailed cost component analysis
- **Statistical Significance**: A/B testing framework

## 🔧 Configuration Management

YAML-based configuration for easy experimentation:

```yaml
# configs/traditional.yaml
EOQ:
  holding_cost: 2.0
  ordering_cost: 50.0
  service_level: 0.95

Safety_Stock:
  service_level: 0.95
  lead_time: 7
  review_period: 1
```

This architecture provides:
- **Modularity**: Easy to add new methods
- **Consistency**: Unified interfaces across all approaches
- **Scalability**: JAX backend for high performance
- **Extensibility**: Clear extension points for new features
- **Reproducibility**: Configuration-driven experiments