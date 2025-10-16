# Distributed Training Experiments

This directory contains scripts for testing distributed training capabilities using JAX.

## distributed_training.py

A comprehensive experiment script that tests different distributed training strategies with profiling support.

### Features

- **Multiple Parallelism Strategies**: Auto, data parallel, model parallel, and hybrid
- **Two Model Types**: Simple RNN and Transformer models
- **GPU Profiling**: Optional performance profiling and benchmarking
- **Synthetic Data Generation**: Generates realistic time series inventory demand data
- **Result Tracking**: Automatically saves experiment results and profiling data to JSON

### Usage

Basic usage with auto strategy selection:
```bash
python experiments/distributed_training.py --strategy auto
```

Enable profiling:
```bash
python experiments/distributed_training.py --strategy auto --profile
```

Use Transformer model with custom parameters:
```bash
python experiments/distributed_training.py \
    --model transformer \
    --strategy data \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.0001 \
    --profile
```

Test different parallelism strategies:
```bash
# Data parallelism
python experiments/distributed_training.py --strategy data --profile

# Model parallelism (requires multiple devices)
python experiments/distributed_training.py --strategy model --profile

# Hybrid parallelism (requires multiple devices)
python experiments/distributed_training.py --strategy hybrid --profile
```

### Command-Line Arguments

- `--strategy`: Parallelism strategy (auto, data, model, hybrid). Default: auto
- `--model`: Model type (simple, transformer). Default: simple
- `--samples`: Number of training samples. Default: 1000
- `--seq-len`: Sequence length for time series. Default: 50
- `--batch-size`: Global batch size. Default: 32
- `--epochs`: Number of training epochs. Default: 20
- `--lr`: Learning rate. Default: 0.001
- `--profile`: Enable GPU profiling
- `--no-save`: Don't save results to file
- `--seed`: Random seed. Default: 42

### Output

Results are saved to `results/` directory:
- `distributed_training_TIMESTAMP.json`: Full experiment results including training history
- `profile_TIMESTAMP.json`: Detailed profiling data (if --profile is used)

### Example Output

```
================================================================================
RESULTS
================================================================================
Total training time: 1.21s
Time per epoch: 0.06s
Final train loss: 74.218750
Initial train loss: 557.464844
Loss reduction: 86.69%
Final eval loss: 26.786945
Best eval loss: 16.256037
```

### System Requirements

- JAX with CPU or GPU support
- Flax for neural networks
- Optax for optimization
- NumPy for data generation

For multi-device distributed training (model/hybrid strategies), ensure you have:
- Multiple GPUs/TPUs available
- Proper JAX distributed setup
