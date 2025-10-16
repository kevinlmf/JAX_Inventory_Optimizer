# Examples Guide

Complete, runnable examples demonstrating all features of the JAX Inventory Optimizer.

## Overview

This directory contains progressive examples that teach you how to use the inventory optimization platform, from basic concepts to advanced enterprise features.

## Quick Start

```bash
# Run all examples in sequence
for script in examples/*.py; do
    echo "Running $script..."
    python "$script"
done

# Or run individual examples
python examples/01_basic_usage.py
python examples/02_ml_forecasting.py
python examples/03_cost_optimization.py
python examples/04_api_client.py
```

## Example Files

### 01_basic_usage.py
**Getting Started with Traditional Methods**

Learn the fundamentals:
- Creating inventory states
- Using Economic Order Quantity (EOQ)
- Safety stock calculations
- (s,S) policy implementation
- Method comparison
- Simple time-series simulation

**Run time:** ~5 seconds
**Prerequisites:** None

```bash
python examples/01_basic_usage.py
```

**What you'll learn:**
- How to initialize optimization methods
- How to make ordering decisions
- How to interpret recommendations
- Basic cost calculations
- When to use each traditional method

---

### 02_ml_forecasting.py
**Machine Learning for Demand Forecasting**

Advanced forecasting with neural networks:
- Training LSTM models
- Making demand forecasts
- Comparing ML vs traditional methods
- Handling complex demand patterns
- Model size comparison

**Run time:** ~30-60 seconds (includes model training)
**Prerequisites:** JAX, Flax

```bash
python examples/02_ml_forecasting.py
```

**What you'll learn:**
- How to train LSTM models on historical data
- When ML outperforms traditional methods
- How to handle non-stationary demand
- Model configuration trade-offs
- Forecasting accuracy metrics

---

### 03_cost_optimization.py
**Enterprise Cost Optimization Features**

Production-grade cost analysis:
- JAX JIT-accelerated optimization (50-100x speedup)
- Deadstock detection and recovery
- Multi-strategy optimization
- Cash flow forecasting
- Working capital analysis
- Automated decision engine

**Run time:** ~10 seconds
**Prerequisites:** JAX, pandas

```bash
python examples/03_cost_optimization.py
```

**What you'll learn:**
- How to leverage JAX for massive speedups
- How to identify and recover deadstock
- How to optimize across multiple strategies
- How to forecast cash flow requirements
- How to analyze working capital efficiency
- How to use automated decision-making

**Key features demonstrated:**
- **JIT Optimization:** Process 100s of SKUs in milliseconds
- **Deadstock Detection:** Multi-dimensional risk scoring
- **Cash Flow Forecasting:** Monte Carlo simulation with confidence intervals
- **Working Capital:** Turnover ratios and efficiency metrics
- **Auto Decisions:** Confidence-scored recommendations

---

### 04_api_client.py
**REST API Integration**

Production API usage:
- Health checks and monitoring
- Single model recommendations
- Batch model comparison
- ML-based forecasting via API
- Error handling
- Performance testing

**Run time:** ~5 seconds
**Prerequisites:** FastAPI server running

```bash
# Terminal 1: Start API server
uvicorn src.api.main:app --reload

# Terminal 2: Run client examples
python examples/04_api_client.py
```

**What you'll learn:**
- How to interact with the REST API
- How to request recommendations
- How to compare multiple models
- How to handle errors gracefully
- API performance characteristics

---

## Learning Path

### Beginn (Start here!)
1. **01_basic_usage.py** - Understand core concepts
2. **04_api_client.py** - Learn API integration

### Intermediate
3. **02_ml_forecasting.py** - Add ML forecasting
4. **03_cost_optimization.py** - Optimize costs

### Advanced
- Combine techniques from multiple examples
- Modify examples for your specific use case
- Integrate with your ERP/inventory system

## Common Use Cases

### Retail / E-commerce
```bash
# Focus on ML forecasting and cost optimization
python examples/02_ml_forecasting.py
python examples/03_cost_optimization.py
```

### Manufacturing
```bash
# Start with traditional methods
python examples/01_basic_usage.py
# Add deadstock detection
python examples/03_cost_optimization.py
```

### Distribution / Wholesale
```bash
# Multi-SKU portfolio optimization
python examples/03_cost_optimization.py
# API integration for real-time decisions
python examples/04_api_client.py
```

## Customization Guide

### Adapting Examples to Your Data

#### 1. Replace Sample Data
```python
# Instead of:
portfolio = create_sample_portfolio(num_skus=10)

# Use your data:
portfolio = pd.read_csv('your_inventory_data.csv')
```

#### 2. Adjust Cost Parameters
```python
# Customize for your business:
eoq = EOQMethod(
    holding_cost=YOUR_HOLDING_COST,      # Your warehouse costs
    ordering_cost=YOUR_ORDERING_COST,    # Your procurement costs
    unit_cost=YOUR_UNIT_COST            # Your product costs
)
```

#### 3. Configure Service Levels
```python
# Set your target service level:
safety_stock = SafetyStockMethod(
    service_level=0.99,  # 99% for critical items
    # or 0.95 for standard items
    # or 0.90 for low-priority items
)
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root
cd JAX_Inventory_Optimizer

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import jax; print(jax.__version__)"
```

### API Connection Errors
```bash
# Make sure API server is running
uvicorn src.api.main:app --reload

# Check server health
curl http://localhost:8000/health
```

### Performance Issues
```bash
# Check JAX device
python -c "import jax; print(jax.devices())"

# For GPU acceleration, install CUDA-enabled JAX
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Performance Benchmarks

Expected run times on typical hardware:

| Example | CPU (MacBook Pro) | GPU (NVIDIA V100) |
|---------|-------------------|-------------------|
| 01_basic_usage.py | 5s | 3s |
| 02_ml_forecasting.py | 45s | 15s |
| 03_cost_optimization.py | 8s | 2s |
| 04_api_client.py | 5s | 5s |

## Next Steps

After completing these examples:

1. **Run Experiments:** Try `experiments/compare_all_methods.py`
2. **Deploy API:** Use Docker or Kubernetes configs
3. **Integrate:** Connect to your data sources
4. **Customize:** Adapt methods to your business rules
5. **Scale:** Use distributed training for large portfolios

## Additional Resources

- **Full Documentation:** See main [README.md](../README.md)
- **Architecture:** Review [ARCHITECTURE.md](../ARCHITECTURE.md)
- **API Docs:** Run server and visit http://localhost:8000/docs
- **Experiments:** Check `experiments/` directory for research scripts
- **Deployment:** See `docs/DEPLOYMENT.md`

## Getting Help

- Open an issue on GitHub
- Check existing examples for similar patterns
- Review API documentation at `/docs` endpoint
- Read inline code comments for detailed explanations

## Contributing Examples

Have a useful example? We'd love to include it!

1. Follow the existing format
2. Include clear documentation
3. Add error handling
4. Test with sample data
5. Submit a pull request

## Example Output Preview

### Cost Optimization Output
```
JAX INVENTORY OPTIMIZER - COST OPTIMIZATION EXAMPLES
================================================================================

Example 1: JIT-Accelerated Cost Optimization
================================================================================

Portfolio Overview:
  Number of SKUs: 100
  Total inventory value: $156,234.50
  Average inventory: 124.3 units

Optimization Results:
  Current daily holding cost: $856.21
  Optimal daily holding cost: $623.45
  Daily savings: $232.76
  Annual savings: $84,957.40

⚡ Performance Note: JAX JIT compilation provides 50-100x speedup
```

### ML Forecasting Output
```
Example 3: ML vs Traditional Methods
================================================================================

Mean Absolute Error (MAE):
  Traditional (EOQ): 8.23
  LSTM:              5.47
  Improvement:       33.5%

Why LSTM Performs Better:
  - Captures temporal patterns and trends
  - Learns from sequence relationships
  - Adapts to seasonality
  - Better for complex, non-stationary demand
```

## License

These examples are part of the JAX Inventory Optimizer project and are licensed under the MIT License.
