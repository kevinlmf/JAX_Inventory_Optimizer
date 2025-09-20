# JAX Inventory Optimizer

A high-performance inventory optimization system using JAX that compares traditional statistical methods with modern ML/RL approaches for retail e-commerce inventory management.

## Overview

This project addresses the fundamental challenge of inventory management in retail e-commerce by implementing and comparing three classes of optimization methods:

- **Traditional Methods**: EOQ, (s,S) policy, safety stock models
- **Machine Learning Methods**: LSTM, Transformer, XGBoost for demand forecasting
- **Reinforcement Learning Methods**: DQN, PPO, SAC for policy learning

The system is built on JAX for high-performance numerical computing and provides both batch experimentation and real-time API service capabilities.

## Problem Formulation

### Business Context
- **Domain**: Retail e-commerce (fast-moving consumer goods and apparel)
- **Scope**: Single SKU, single warehouse with fixed lead time
- **Objective**: Minimize total inventory cost while maintaining service levels ≥95%

### Mathematical Framework

**State Space:**
```
s_t = (I_t, O_t, H_t)
```
- I_t: Current inventory level
- O_t: Outstanding orders (in-transit)
- H_t: Demand history vector

**Action Space:**
```
a_t ∈ [0, A_max]  # Order quantity
```

**Cost Function:**
```
C_t = h·max(I_t,0) + p·max(-I_t,0) + K·𝟙(a_t>0) + c·a_t
```
- h: Holding cost per unit per period
- p: Stockout penalty per unit
- K: Fixed ordering cost
- c: Variable cost per unit ordered

## Technology Stack

- **Core**: JAX, Flax, Optax for high-performance computing
- **Traditional Models**: SciPy, statsmodels
- **Machine Learning**: TensorFlow, PyTorch (for comparison)
- **API**: FastAPI, Uvicorn
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Plotly, Seaborn

## Project Structure

```
JAX-Inventory-Optimizer/
├── src/
│   ├── core/                   # Core interfaces and utilities
│   ├── methods/                # Algorithm implementations
│   │   ├── traditional/        # EOQ, Safety Stock, (s,S) policy
│   │   ├── ml_methods/         # LSTM, Transformer
│   │   └── rl_methods/         # DQN, PPO, SAC
│   ├── data/                   # Data generation and preprocessing
│   ├── api/                    # FastAPI service
│   └── utils/                  # Shared utilities
├── experiments/                # Experimental scripts
├── notebooks/                  # Jupyter analysis notebooks
├── configs/                    # YAML configuration files
├── data/                       # Generated datasets
├── results/                    # Experimental results
├── examples/                   # Usage examples
└── tests/                      # Unit tests
```

## Installation

```bash
git clone https://github.com/your-username/JAX-Inventory-Optimizer
cd JAX-Inventory-Optimizer
pip install -r requirements.txt
```

## Quick Start

### 1. Initialize Project
```bash
python setup_project.py    # Create project structure and configurations
python create_sample_data.py  # Generate sample retail data (5 stores, 10 items, 3 years)
```

### 2. Run Experiments (Recommended Order)

**Step 1: Basic Verification**
```bash
python examples/quick_start.py  # Quick project health check (~5 seconds)
```

**Step 2: Test Traditional Methods**
```bash
python experiments/test_traditional_methods.py  # Test EOQ, Safety Stock, (s,S) policies (~30 seconds)
```

**Step 3: Full Method Comparison**
```bash
python experiments/compare_all_methods.py  # Compare Traditional + ML + RL methods (~5-15 minutes)
```

### 3. Start API Service
```bash
python src/api/main.py  # REST API service for real-time recommendations
```
API available at http://localhost:8000 with interactive docs at /docs

### 4. Explore with Jupyter
```bash
jupyter notebook notebooks/quick_start.ipynb
```

## Script Comparison

| Script | Purpose | Runtime | Methods Tested | Output |
|--------|---------|---------|----------------|--------|
| `examples/quick_start.py` | Project health check | 5s | None (demo only) | Console messages |
| `experiments/test_traditional_methods.py` | Traditional methods validation | 30s | 8 traditional methods | Performance comparison |
| `experiments/compare_all_methods.py` | Full method comparison | 5-15min | All (Traditional + ML + RL) | CSV reports + visualizations |
| `src/api/main.py` | Production API service | Continuous | All available | REST API endpoints |

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Inventory Recommendation
```bash
curl -X POST "http://localhost:8000/recommend/eoq" \
-H "Content-Type: application/json" \
-d '{
  "current_inventory": 100,
  "outstanding_orders": 20,
  "demand_history": [10, 15, 12, 18, 14, 16, 11],
  "forecast_horizon": 7
}'
```

### Compare Multiple Models
```bash
curl -X POST "http://localhost:8000/batch_recommend" \
-H "Content-Type: application/json" \
-d '{
  "models": ["eoq", "safety_stock"],
  "current_inventory": 100,
  "outstanding_orders": 20,
  "demand_history": [10, 15, 12, 18, 14, 16, 11]
}'
```

## Configuration

The system uses YAML configuration files in `configs/`:

- `inventory.yaml`: Inventory problem parameters
- `traditional.yaml`: Traditional method settings
- `ml.yaml`: ML method hyperparameters
- `rl.yaml`: RL algorithm configuration
- `experiment.yaml`: Experiment settings

## Performance Metrics

**Primary KPIs:**
- Total cost (holding + stockout + ordering)
- Service level (demand fulfillment rate)
- Inventory turnover ratio

**Efficiency Metrics:**
- Forecast accuracy (MAPE, RMSE)
- Computational speed (JAX JIT advantage)
- Model adaptability to demand changes

## Experiments

Run comparative experiments:
```bash
python experiments/compare_all_methods.py
python experiments/test_traditional_methods.py
```

Results are saved in the `results/` directory with detailed performance analysis.

## Research Objectives

1. **Performance Quantification**: Measure improvement of ML/RL over traditional methods
2. **Computational Efficiency**: Leverage JAX for real-time decision making
3. **Practical Deployment**: Bridge research-to-production gap
4. **Robustness Analysis**: Test under various demand scenarios

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

