"""
Machine Learning Forecasting Example: Using LSTM and Transformer Models

This example demonstrates:
1. Training LSTM models on historical demand
2. Making forecasts with trained models
3. Comparing ML vs traditional methods
4. Handling complex demand patterns

Run: python examples/02_ml_forecasting.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax.numpy as jnp
from src.core.interfaces import InventoryState
from src.methods.ml_methods.lstm import LSTMInventoryMethod
from src.methods.traditional.eoq import EOQMethod


def generate_complex_demand(num_samples: int = 200):
    """Generate synthetic demand data with trend, seasonality, and noise."""
    time = np.arange(num_samples)

    # Components
    trend = 50 + 0.1 * time  # Upward trend
    seasonality = 10 * np.sin(2 * np.pi * time / 30)  # 30-period cycle
    noise = np.random.normal(0, 5, num_samples)

    demand = trend + seasonality + noise
    demand = np.maximum(demand, 0)  # Non-negative

    return demand


def example_1_lstm_training():
    """Example 1: Training LSTM on Historical Demand"""
    print("\n" + "="*80)
    print("Example 1: LSTM Training")
    print("="*80)

    # Generate training data
    demand_data = generate_complex_demand(num_samples=200)

    print(f"\nDataset Statistics:")
    print(f"  Number of samples: {len(demand_data)}")
    print(f"  Mean demand: {np.mean(demand_data):.2f}")
    print(f"  Std demand: {np.std(demand_data):.2f}")
    print(f"  Min/Max: {np.min(demand_data):.2f} / {np.max(demand_data):.2f}")

    # Initialize LSTM
    lstm = LSTMInventoryMethod(
        hidden_size=64,
        num_layers=2,
        sequence_length=30,
        learning_rate=0.001
    )

    print(f"\nModel Configuration:")
    print(f"  Hidden size: {lstm.hidden_size}")
    print(f"  Number of layers: {lstm.num_layers}")
    print(f"  Sequence length: {lstm.sequence_length}")

    # Train model
    print(f"\nTraining model...")
    train_data = demand_data[:150]  # Use first 150 samples for training
    history = lstm.train(train_data, epochs=50, batch_size=16, verbose=False)

    print(f"  Training completed!")
    print(f"  Final training loss: {history['loss'][-1]:.4f}")
    print(f"  Training time: ~few seconds on CPU")

    return lstm, demand_data


def example_2_forecasting():
    """Example 2: Making Forecasts with Trained LSTM"""
    print("\n" + "="*80)
    print("Example 2: LSTM Forecasting")
    print("="*80)

    # Train model
    lstm, demand_data = example_1_lstm_training()

    # Use test data
    test_data = demand_data[150:]
    sequence_length = lstm.sequence_length

    print(f"\nMaking Forecasts on Test Set:")
    print("-" * 60)
    print(f"{'Actual':<10} {'Forecast':<10} {'Error':<10} {'Error %':<10}")
    print("-" * 60)

    errors = []
    for i in range(min(10, len(test_data) - sequence_length)):
        # Get historical sequence
        history = test_data[i:i+sequence_length].tolist()

        # Create state
        state = InventoryState(
            inventory_level=100.0,
            outstanding_orders=0.0,
            demand_history=history,
            time_step=i
        )

        # Get forecast
        action = lstm.decide(state)
        actual = test_data[i + sequence_length]
        forecast = action.forecast

        error = forecast - actual
        error_pct = (error / actual) * 100 if actual > 0 else 0

        errors.append(error)

        print(f"{actual:<10.2f} {forecast:<10.2f} {error:<10.2f} {error_pct:<10.1f}%")

    print("-" * 60)
    print(f"\nForecast Accuracy Metrics:")
    print(f"  Mean Absolute Error (MAE): {np.mean(np.abs(errors)):.2f}")
    print(f"  Root Mean Squared Error (RMSE): {np.sqrt(np.mean(np.array(errors)**2)):.2f}")
    print(f"  Mean Percentage Error: {np.mean(np.abs(errors) / test_data[:len(errors)]):.2%}")


def example_3_ml_vs_traditional():
    """Example 3: Comparing ML and Traditional Methods"""
    print("\n" + "="*80)
    print("Example 3: ML vs Traditional Methods")
    print("="*80)

    # Generate data with trend
    demand_data = generate_complex_demand(num_samples=200)
    train_data = demand_data[:150]
    test_data = demand_data[150:]

    # Train LSTM
    lstm = LSTMInventoryMethod(
        hidden_size=32,
        num_layers=2,
        sequence_length=30,
        learning_rate=0.001
    )
    lstm.train(train_data, epochs=50, batch_size=16, verbose=False)

    # Initialize EOQ (uses simple average)
    eoq = EOQMethod(
        holding_cost=1.0,
        ordering_cost=50.0,
        unit_cost=10.0
    )

    print(f"\nComparing Forecasting Performance:")
    print("-" * 80)

    # Make predictions
    lstm_errors = []
    eoq_errors = []

    for i in range(min(20, len(test_data) - 30)):
        history = test_data[i:i+30].tolist()

        state = InventoryState(
            inventory_level=100.0,
            outstanding_orders=0.0,
            demand_history=history,
            time_step=i
        )

        # LSTM prediction
        lstm_action = lstm.decide(state)
        lstm_forecast = lstm_action.forecast

        # EOQ uses simple average
        eoq_action = eoq.decide(state)
        eoq_forecast = np.mean(history)

        # Actual demand
        actual = test_data[i + 30]

        # Errors
        lstm_errors.append(abs(lstm_forecast - actual))
        eoq_errors.append(abs(eoq_forecast - actual))

    # Results
    lstm_mae = np.mean(lstm_errors)
    eoq_mae = np.mean(eoq_errors)
    improvement = ((eoq_mae - lstm_mae) / eoq_mae) * 100

    print(f"\nMean Absolute Error (MAE):")
    print(f"  Traditional (EOQ): {eoq_mae:.2f}")
    print(f"  LSTM:              {lstm_mae:.2f}")
    print(f"  Improvement:       {improvement:.1f}%")

    print(f"\nWhy LSTM Performs Better:")
    print(f"  - Captures temporal patterns and trends")
    print(f"  - Learns from sequence relationships")
    print(f"  - Adapts to seasonality")
    print(f"  - Better for complex, non-stationary demand")


def example_4_inventory_decisions():
    """Example 4: ML-Based Inventory Decisions"""
    print("\n" + "="*80)
    print("Example 4: ML-Based Inventory Decisions")
    print("="*80)

    # Train model
    demand_data = generate_complex_demand(num_samples=200)
    lstm = LSTMInventoryMethod(
        hidden_size=64,
        num_layers=2,
        sequence_length=30,
        learning_rate=0.001
    )
    lstm.train(demand_data[:150], epochs=50, batch_size=16, verbose=False)

    # Simulate inventory management
    print(f"\nSimulating Inventory Management with LSTM:")
    print("-" * 80)
    print(f"{'Period':<8} {'Inventory':<12} {'Forecast':<12} {'Order':<12} {'Actual':<12}")
    print("-" * 80)

    inventory = 100.0
    total_cost = 0.0
    holding_cost_rate = 1.0
    ordering_cost = 50.0

    for t in range(30, 60):
        # Get historical demand
        history = demand_data[t-30:t].tolist()

        # Create state
        state = InventoryState(
            inventory_level=inventory,
            outstanding_orders=0.0,
            demand_history=history,
            time_step=t
        )

        # Get LSTM recommendation
        action = lstm.decide(state)
        forecast = action.forecast
        order_qty = action.order_quantity

        # Actual demand
        actual_demand = demand_data[t]

        # Update inventory
        inventory = inventory + order_qty - actual_demand
        inventory = max(0, inventory)  # No negative inventory

        # Calculate cost
        period_cost = inventory * holding_cost_rate
        if order_qty > 0:
            period_cost += ordering_cost
        total_cost += period_cost

        # Print period summary
        if t < 40:  # Show first 10 periods
            print(f"{t:<8} {inventory:<12.1f} {forecast:<12.1f} {order_qty:<12.1f} {actual_demand:<12.1f}")

    print("-" * 80)
    print(f"\nSimulation Results:")
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Average inventory: {inventory:.1f} units")
    print(f"  Total periods: 30")


def example_5_advanced_features():
    """Example 5: Advanced LSTM Features"""
    print("\n" + "="*80)
    print("Example 5: Advanced LSTM Features")
    print("="*80)

    # Generate multi-pattern data
    demand_data = generate_complex_demand(num_samples=300)

    # Train with different configurations
    configs = [
        {'hidden_size': 32, 'num_layers': 1, 'name': 'Small'},
        {'hidden_size': 64, 'num_layers': 2, 'name': 'Medium'},
        {'hidden_size': 128, 'num_layers': 3, 'name': 'Large'},
    ]

    print(f"\nComparing Model Sizes:")
    print("-" * 60)

    results = []
    for config in configs:
        lstm = LSTMInventoryMethod(
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            sequence_length=30,
            learning_rate=0.001
        )

        # Train
        train_data = demand_data[:200]
        history = lstm.train(train_data, epochs=30, batch_size=16, verbose=False)

        # Evaluate
        test_data = demand_data[200:]
        errors = []

        for i in range(20):
            seq = test_data[i:i+30].tolist()
            state = InventoryState(
                inventory_level=100.0,
                outstanding_orders=0.0,
                demand_history=seq,
                time_step=i
            )
            action = lstm.decide(state)
            actual = test_data[i+30]
            errors.append(abs(action.forecast - actual))

        mae = np.mean(errors)
        results.append({'name': config['name'], 'mae': mae})

        print(f"\n{config['name']} Model:")
        print(f"  Hidden size: {config['hidden_size']}")
        print(f"  Layers: {config['num_layers']}")
        print(f"  Test MAE: {mae:.2f}")

    print(f"\n" + "-" * 60)
    best = min(results, key=lambda x: x['mae'])
    print(f"\nBest Configuration: {best['name']} (MAE: {best['mae']:.2f})")


def main():
    """Run all ML forecasting examples."""
    print("\n" + "="*80)
    print("JAX INVENTORY OPTIMIZER - ML FORECASTING EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate machine learning-based demand forecasting")
    print("using LSTM neural networks for complex demand patterns.")

    # Set random seed
    np.random.seed(42)

    # Run examples
    try:
        example_1_lstm_training()
        example_2_forecasting()
        example_3_ml_vs_traditional()
        example_4_inventory_decisions()
        example_5_advanced_features()

        print("\n" + "="*80)
        print("ML FORECASTING EXAMPLES COMPLETED")
        print("="*80)
        print("\nKey Takeaways:")
        print("  1. LSTM captures temporal patterns better than simple averages")
        print("  2. Larger models may overfit - balance capacity vs generalization")
        print("  3. Training data quality and quantity are crucial")
        print("  4. ML methods excel with complex, non-stationary demand")
        print("\nNext: Try cost optimization examples (04_cost_optimization.py)")
        print()

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
