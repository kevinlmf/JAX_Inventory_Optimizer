# JAX Inventory Optimizer  

A **high-performance inventory optimization system powered by JAX**, unifying **traditional methods**, **machine learning**, and **reinforcement learning** under a single stochastic control framework for retail e-commerce.  

## Overview  

This project is built on **JAX**, which provides:  
- **Automatic differentiation** for gradient-based optimization,  
- **JIT compilation** for near-C++ level execution speed,  
- **Vectorization (`vmap`)** and **parallelization (`pmap`)** for large-scale Monte Carlo simulations.  

These capabilities allow us to:  
- Scale from toy problems to **industry-sized datasets**,  
- Run **thousands of experiments in parallel** for policy evaluation,  
- Seamlessly integrate **EOQ policies, ML demand forecasting, and RL decision-making** in one unified framework.  

By leveraging JAX, the system bridges the gap between **classical inventory theory** and **modern AI-driven methods**, achieving both research flexibility and production-level performance.  

## Problem Formulation  

### Business Context  
- **Domain**: Retail e-commerce (fast-moving consumer goods and apparel)  
- **Scope**: Single SKU, single warehouse with fixed lead time  
- **Objective**: Minimize total inventory cost while maintaining service levels ≥ 95%  

### Mathematical Framework  

**State Space**  
```
s_t = (I_t, O_t, H_t)
```  
- `I_t`: current inventory level  
- `O_t`: outstanding orders (in-transit)  
- `H_t`: demand history vector  

**Action Space**  
```
a_t ∈ [0, A_max]    # order quantity at time t
```  

**Cost Function**  
```
C_t = h * max(I_t, 0)
    + p * max(-I_t, 0)
    + K * 1(a_t > 0)
    + c * a_t
```  
- `h`: holding cost per unit per period  
- `p`: stockout penalty per unit  
- `K`: fixed ordering cost  
- `c`: variable cost per unit ordered  

## Optimization Methods  

- **Traditional**: EOQ, (s,S) policy, safety stock models  
- **Machine Learning**: LSTM, Transformer, XGBoost for demand forecasting  
- **Reinforcement Learning**: DQN, PPO, SAC for policy learning  


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
# clone
git clone https://github.com/kevinlmf/JAX_Inventory_Optimizer.git
cd JAX_Inventory_Optimizer

# install dependencies
pip install -r requirements.txt
```  

## Quick Start  

**Step 1: Health Check**  
```bash
python examples/quick_start.py
```  

**Step 2: Run Traditional Methods**  
```bash
python experiments/test_traditional_methods.py
```  

**Step 3: Compare All Methods**  
```bash
python experiments/compare_all_methods.py
```  

**Step 4: Start API Service**  
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```  
- API: [http://localhost:8000](http://localhost:8000)  
- Docs: [http://localhost:8000/docs](http://localhost:8000/docs)  

## Performance Metrics  

- **Primary KPIs**: Total cost, service level, turnover ratio  
- **Forecast Metrics**: MAPE, RMSE  
- **RL Metrics**: Mean reward, regret, policy robustness  
- **Efficiency**: Runtime speed (JAX vs NumPy/PyTorch), scalability  

## Research Objectives  

1. **Performance Quantification**: ML/RL vs traditional methods  
2. **Computational Efficiency**: JAX advantages for real-time decision making  
3. **Research-to-Production Bridge**: APIs + configs for deployment  
4. **Robustness**: Stress-testing under volatile demand