# JAX Inventory Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)

> **Enterprise-grade inventory optimization platform powered by JAX for accelerated computation and intelligent decision-making.**

## Overview

JAX Inventory Optimizer addresses the fundamental challenge in supply chain management: minimizing inventory costs while maintaining service levels through hardware-accelerated computation and intelligent algorithms.

---

## Core Features

### High-Performance Computing
- **JAX-Accelerated**: JIT compilation, automatic differentiation, GPU/TPU support
- **Distributed Training**: Data/model/hybrid parallelism (7.2x speedup on 8 GPUs)
- **Vectorized Operations**: Efficient batch processing across multi-SKU portfolios

### Multi-Paradigm Optimization
- **Traditional Methods**: EOQ, Safety Stock, (s,S) Policy
- **Machine Learning**: LSTM with Attention, Transformer architectures
- **Reinforcement Learning**: Deep Q-Network (DQN) for dynamic control

### Cost Optimization Suite
- **Dynamic Inventory Optimization**: Strategy-based (Aggressive/Balanced/Conservative)
- **Deadstock Detection**: Multi-dimensional risk assessment
- **Cash Flow Prediction**: Monte Carlo simulation with confidence intervals
- **Portfolio Optimization**: Budget constraints and capital allocation
- **Auto Decision Engine**: Automated inventory management with risk controls

### Production-Ready
- **FastAPI REST Service**  
  Exposes real-time endpoints for **inventory forecasting**, **reorder recommendations**, and **policy evaluation**.  
  The service layer is fully asynchronous and supports **batch inference** via JAX-compiled models, enabling **sub-10 ms latency** per request under GPU acceleration.

- **Docker & Kubernetes Deployment**  
  All components (API, training workers, data pipeline) are fully **containerized** with reproducible builds.  
  Kubernetes manifests and Helm charts provide **scalable**, **fault-tolerant**, and **auto-healing** deployments across environments (local, staging, and cloud).  
  The system supports **rolling updates**, **horizontal pod autoscaling**, and integrates **Prometheus + Grafana** for end-to-end observability.

- **MLOps Integration**  
  Integrated with **Weights & Biases (W&B)** for unified **experiment tracking**, **hyperparameter logging**, and **model performance visualization**.  
  Automatically syncs metrics, configurations, and artifacts, ensuring **reproducible training pipelines** and seamless **CI/CD** compatibility (e.g., GitHub Actions, Vertex AI).

---
##  From Research to Production: The End-to-End ML Lifecycle

We can divide the machine learning system lifecycle into **five key stages** :

| Stage | Main Objective | Responsible Role | Tools / Frameworks |
|:------:|----------------|------------------|--------------------|
| **1️⃣ Data Preparation (Data)** | Collect, clean, and label data | Data Engineer | `pandas`, `Airflow`, `SQL` |
| **2️⃣ Model Training (Modeling)** | Build, tune, and validate models | Researcher / Data Scientist | `PyTorch`, `JAX`, `sklearn` |
| **3️⃣ Evaluation & Optimization (Evaluation)** | Compare experiments and select the best model | ML Engineer | `Weights & Biases (W&B)`, `MLflow` |
| **4️⃣ Deployment (Deployment)** | Package the model into a scalable API service | DevOps Engineer | `Docker`, `FastAPI` |
| **5️⃣ Monitoring & Maintenance (MLOps)** | Automate retraining, monitor drift, and manage versions | MLOps Engineer | `Kubernetes`, `Prometheus`, `CI/CD` |

---

 **Steps 1–3:** Make the model *run*  
 **Step 4:** Make the model *usable in production*  
 **Step 5 (MLOps):** Make the model *sustain itself and evolve continuously*
---

## Quick Start

### Installation
```bash
git clone https://github.com/kevinlmf/JAX_Inventory_Optimizer
cd JAX_Inventory_Optimizer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Verify Setup
```bash
python -c "import jax; print(f'JAX version: {jax.__version__}')"
python -c "import jax; print(f'Devices: {jax.devices()}')"
```

### Run Demo
```bash
# Run all demos
./run_all_demos.sh

# Start API server
uvicorn src.api.main:app --reload
```

---

## Usage Examples

### Run Demo Scripts

```bash
# Compare all optimization methods (Traditional + ML + RL)
python experiments/compare_all_methods.py
# Output: results/comparisons/method_comparison_*.csv + performance plots

# Enterprise features: Cost optimization + Risk management
python experiments/demo_enterprise.py
# Features: Deadstock detection, cash flow prediction, anomaly detection

# API service demonstration
python experiments/demo_api.py
# Test: Model recommendations, batch processing, health checks

# MLOps: Experiment tracking + GPU profiling
python experiments/demo_mlops.py
# Integration: Weights & Biases tracking, performance profiling

# Distributed training (requires multiple GPUs)
python experiments/distributed_training.py --strategy auto --profile
# Strategies: data/model/hybrid parallelism
```

### Hands-on Examples

```bash
# 01: Basic usage - Traditional methods (EOQ, Safety Stock, s-S)
python examples/01_basic_usage.py
# Learn: Core concepts, inventory states, ordering decisions

# 02: ML forecasting - LSTM demand prediction
python examples/02_ml_forecasting.py
# Learn: Neural network training, complex pattern recognition

# 03: Cost optimization - Enterprise features
python examples/03_cost_optimization.py
# Features: JIT optimization (50-100x speedup), deadstock detection, cash flow

# 04: API client - Production integration
python examples/04_api_client.py
# Requires: uvicorn src.api.main:app --reload (in another terminal)
```

See [examples/README.md](examples/README.md) for detailed documentation.

---

## Architecture

```
JAX_Inventory_Optimizer/
├── src/
│   ├── core/                    # Core interfaces and framework
│   ├── methods/                 # Optimization algorithms
│   │   ├── traditional/         # EOQ, Safety Stock, (s,S)
│   │   ├── ml_methods/          # LSTM, Transformer
│   │   └── rl_methods/          # DQN agent
│   ├── cost_optimization/       # Financial analytics
│   ├── risk_management/         # Anomaly detection
│   ├── distributed/             # Multi-GPU training
│   ├── api/                     # FastAPI service
│   └── data/                    # Data management
├── experiments/                 # Demo scripts
├── k8s/                        # Kubernetes manifests
├── helm/                       # Helm charts
└── requirements.txt
```

---

## Performance Metrics

| Metric | Value | Context |
|--------|-------|---------|
| Computation Speed | 10-100x faster | JAX vs NumPy (GPU) |
| JIT Optimization | 50-100x faster | Portfolio optimization |
| Inference Latency | < 10ms | Per recommendation |
| Distributed Scaling | 7.2x on 8 GPUs | Data parallelism |
| Cost Reduction | 20-35% | Real-world scenarios |

---

## Method Comparison

| Method | Training Time | Inference | Adaptability | Best For |
|--------|--------------|-----------|--------------|----------|
| **EOQ** | None | < 1ms | Static | Stable demand |
| **Safety Stock** | None | < 1ms | Medium | Service targets |
| **LSTM** | 1-5 min | < 10ms | High | Complex patterns |
| **DQN** | 10-60 min | < 5ms | Very High | Dynamic environments |

---

## Docker Deployment

```bash
# CPU container
docker build -t jax-inventory-optimizer .
docker run -p 8000:8000 jax-inventory-optimizer

# GPU container
docker build -f Dockerfile.gpu -t jax-inventory-optimizer:gpu .
docker run --gpus all -p 8000:8000 jax-inventory-optimizer:gpu

# Docker Compose
docker-compose up
```

---

## Kubernetes Deployment

```bash
# Deploy with Helm
helm install inventory-optimizer ./helm/jax-optimizer

# Or raw manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -l app=inventory-optimizer
```

---

## Advanced Features

### Distributed Training
```bash
python experiments/distributed_training.py --strategy auto --profile
```

### Anomaly Detection
```python
from src.risk_management.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(method='zscore', threshold=3.0)
anomalies = detector.detect(demand_history)
```

### Cash Flow Forecasting
```python
from src.cost_optimization.cashflow_predictor import CashFlowPredictor

predictor = CashFlowPredictor()
forecast = predictor.predict_cashflow(
    inventory_levels=[100, 200],
    demand_forecasts=[50, 80],
    forecast_horizon=90
)
```





