"""
Comprehensive Comparison of All Inventory Methods

This script compares Traditional, ML, and RL approaches on the same dataset
and provides detailed performance analysis and visualizations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import methods
from methods.traditional.eoq import EOQMethod
from methods.traditional.safety_stock import SafetyStockMethod, AdaptiveSafetyStock
from methods.traditional.s_S_policy import sSPolicyMethod
from comparison.evaluator import EnhancedInventoryEvaluator
from core.interfaces import InventoryState

# ML and RL imports with error handling
try:
    from methods.ml_methods.lstm import LSTMInventoryMethod
    LSTM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  LSTM not available: {e}")
    LSTM_AVAILABLE = False

try:
    from methods.ml_methods.transformer import TransformerInventoryMethod
    TRANSFORMER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Transformer not available: {e}")
    TRANSFORMER_AVAILABLE = False

try:
    from methods.rl_methods.dqn import DQNInventoryMethod
    DQN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  DQN not available: {e}")
    DQN_AVAILABLE = False


def load_sample_data():
    """Load sample retail data"""
    data_path = Path(__file__).parent.parent / 'data' / 'sample_retail_data.csv'

    if not data_path.exists():
        print("❌ Sample data not found. Please run create_sample_data.py first.")
        return None, None

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])

    # Extract single SKU data (Store 1, Item 1)
    sku_data = df[(df['store'] == 1) & (df['item'] == 1)].copy()
    sku_data = sku_data.sort_values('date')

    return sku_data['sales'].values, sku_data


def create_test_states(train_data: np.ndarray, test_data: np.ndarray, window_size: int = 30):
    """Create InventoryState objects for evaluation"""
    states = []

    for i in range(len(test_data)):
        # Use last window_size points from training + previous test points
        if i == 0:
            history = train_data[-window_size:]
        else:
            combined = np.concatenate([train_data, test_data[:i]])
            history = combined[-window_size:]

        state = InventoryState(
            inventory_level=50.0,  # Starting inventory
            outstanding_orders=0.0,
            demand_history=history,
            time_step=len(train_data) + i
        )
        states.append(state)

    return states


def initialize_traditional_methods():
    """Initialize all traditional methods"""
    methods = {}

    try:
        methods['EOQ'] = EOQMethod(
            holding_cost=1.0,
            ordering_cost=25.0,
            service_level=0.95
        )
        print("  ✅ EOQ initialized")
    except Exception as e:
        print(f"  ❌ EOQ failed: {e}")

    try:
        methods['SafetyStock_Normal'] = SafetyStockMethod(
            service_level=0.95,
            lead_time=7,
            method='normal'
        )
        print("  ✅ Safety Stock (Normal) initialized")
    except Exception as e:
        print(f"  ❌ Safety Stock (Normal) failed: {e}")

    try:
        methods['SafetyStock_Empirical'] = SafetyStockMethod(
            service_level=0.95,
            lead_time=7,
            method='empirical'
        )
        print("  ✅ Safety Stock (Empirical) initialized")
    except Exception as e:
        print(f"  ❌ Safety Stock (Empirical) failed: {e}")

    try:
        methods['AdaptiveSafetyStock'] = AdaptiveSafetyStock(
            service_level=0.95,
            window_size=30
        )
        print("  ✅ Adaptive Safety Stock initialized")
    except Exception as e:
        print(f"  ❌ Adaptive Safety Stock failed: {e}")

    try:
        methods['sS_Policy'] = sSPolicyMethod(
            holding_cost=1.0,
            shortage_cost=5.0,
            ordering_cost=25.0,
            optimization_method='analytical'
        )
        print("  ✅ (s,S) Policy initialized")
    except Exception as e:
        print(f"  ❌ (s,S) Policy failed: {e}")

    return methods


def initialize_ml_methods():
    """Initialize ML methods if available"""
    methods = {}

    if LSTM_AVAILABLE:
        try:
            methods['LSTM_Basic'] = LSTMInventoryMethod(
                sequence_length=20,
                hidden_size=16,
                epochs=5,  # Very fast for demo
                batch_size=32,
                use_attention=False
            )
            print("  ✅ Basic LSTM initialized")
        except Exception as e:
            print(f"  ❌ Basic LSTM failed: {e}")

        try:
            methods['LSTM_Attention'] = LSTMInventoryMethod(
                sequence_length=20,
                hidden_size=16,
                epochs=5,  # Very fast for demo
                batch_size=32,
                use_attention=True
            )
            print("  ✅ Attention LSTM initialized")
        except Exception as e:
            print(f"  ❌ Attention LSTM failed: {e}")

    if TRANSFORMER_AVAILABLE:
        try:
            methods['Transformer'] = TransformerInventoryMethod(
                sequence_length=20,
                d_model=16,  # Very small for demo
                num_heads=2,
                num_layers=1,
                epochs=3,  # Very fast for demo
                batch_size=32
            )
            print("  ✅ Transformer initialized")
        except Exception as e:
            print(f"  ❌ Transformer failed: {e}")

    return methods


def initialize_rl_methods():
    """Initialize RL methods if available"""
    methods = {}

    if DQN_AVAILABLE:
        try:
            methods['DQN'] = DQNInventoryMethod(
                state_dim=6,
                num_actions=21,
                hidden_sizes=(32, 32),  # Very small for demo
                epsilon_decay=0.95,
                memory_size=500  # Smaller for faster training
            )
            print("  ✅ DQN initialized")
        except Exception as e:
            print(f"  ❌ DQN failed: {e}")

    return methods


def fit_methods(methods: dict, train_data: np.ndarray):
    """Fit all methods on training data"""
    fitted_methods = {}

    print(f"🔧 FITTING METHODS ON {len(train_data)} TRAINING SAMPLES")
    print("-" * 60)

    for name, method in methods.items():
        print(f"  Training {name}...")
        start_time = time.time()

        try:
            method.fit(train_data)
            fitted_methods[name] = method
            elapsed = time.time() - start_time
            print(f"  ✅ {name}: {elapsed:.2f}s")

        except Exception as e:
            print(f"  ❌ {name}: {e}")

    # Special handling for RL methods (need additional training)
    for name, method in fitted_methods.items():
        if hasattr(method, 'train_agent') and name in ['DQN']:
            print(f"  🎮 Training RL agent {name}...")
            start_time = time.time()
            try:
                method.train_agent(num_episodes=20)  # Reduced for demo
                elapsed = time.time() - start_time
                print(f"  ✅ {name} RL training: {elapsed:.2f}s")
            except Exception as e:
                print(f"  ❌ {name} RL training: {e}")

    return fitted_methods


def run_comprehensive_comparison():
    """Run comprehensive comparison of all methods"""
    print("🏭 COMPREHENSIVE INVENTORY METHOD COMPARISON")
    print("=" * 60)

    # Load data
    print("📊 Loading data...")
    demand_data, df_full = load_sample_data()

    if demand_data is None:
        return

    # Split data
    train_size = int(0.8 * len(demand_data))
    train_data = demand_data[:train_size]
    test_data = demand_data[train_size:]

    print(f"  Train periods: {len(train_data)}")
    print(f"  Test periods: {len(test_data)}")
    print(f"  Train mean demand: {np.mean(train_data):.2f}")
    print(f"  Train std demand: {np.std(train_data):.2f}")

    # Initialize methods
    print(f"\n🔧 INITIALIZING METHODS")
    print("-" * 60)

    all_methods = {}

    print("📈 Traditional Methods:")
    traditional_methods = initialize_traditional_methods()
    all_methods.update(traditional_methods)

    print("🧠 ML Methods:")
    ml_methods = initialize_ml_methods()
    all_methods.update(ml_methods)

    print("🎮 RL Methods:")
    rl_methods = initialize_rl_methods()
    all_methods.update(rl_methods)

    print(f"\n  Total methods initialized: {len(all_methods)}")

    if not all_methods:
        print("❌ No methods available for comparison")
        return

    # Fit methods
    fitted_methods = fit_methods(all_methods, train_data)

    if not fitted_methods:
        print("❌ No methods successfully fitted")
        return

    # Create test states
    print(f"\n📋 Creating test states...")
    test_states = create_test_states(train_data, test_data)

    # Run evaluation
    print(f"\n🔍 RUNNING COMPREHENSIVE EVALUATION")
    print("-" * 60)

    evaluator = EnhancedInventoryEvaluator(
        holding_cost=1.0,
        stockout_cost=5.0,
        ordering_cost=25.0
    )

    # Compare methods
    results_df = evaluator.compare_methods(fitted_methods, test_states, test_data)

    # Print results
    if not results_df.empty:
        evaluator.print_comparison_summary(results_df, top_n=10)

        # Save detailed results
        output_dir = Path(__file__).parent.parent / 'results' / 'comparisons'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f'method_comparison_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")

        # Create visualization
        try:
            plot_file = output_dir / f'performance_plots_{timestamp}.png'
            evaluator.create_performance_plots(results_df, save_path=str(plot_file))
        except Exception as e:
            print(f"⚠️  Could not create plots: {e}")

        # Method-specific insights
        print_method_insights(results_df, fitted_methods)

    else:
        print("❌ No results to display")

    return results_df


def print_method_insights(results_df: pd.DataFrame, methods: dict):
    """Print insights about method performance"""
    print(f"\n🧠 METHOD INSIGHTS")
    print("-" * 60)

    if results_df.empty:
        return

    # Best in each category
    print("🏆 CATEGORY WINNERS:")
    categories = ['traditional', 'ml', 'rl']

    for category in categories:
        category_methods = results_df[
            results_df['method_category'].str.lower().str.contains(category, na=False)
        ]
        if not category_methods.empty:
            best = category_methods.iloc[0]  # Already sorted by cost
            print(f"  {category.title()}: {best['method_name']} "
                  f"(Cost: ${best['total_cost']:.1f}, Service: {best['service_level']:.1%})")

    # Performance patterns
    print(f"\n📊 PERFORMANCE PATTERNS:")

    # Forecast accuracy correlation
    if 'forecast_accuracy' in results_df.columns:
        corr_service = results_df['forecast_accuracy'].corr(results_df['service_level'])
        corr_cost = results_df['forecast_accuracy'].corr(results_df['total_cost'])
        print(f"  Forecast accuracy vs Service level correlation: {corr_service:.3f}")
        print(f"  Forecast accuracy vs Total cost correlation: {corr_cost:.3f}")

    # Service vs Cost trade-off
    high_service = results_df[results_df['service_level'] >= 0.95]
    if not high_service.empty:
        print(f"  Methods achieving 95%+ service level: {len(high_service)}")
        if len(high_service) > 1:
            best_cost_high_service = high_service.loc[high_service['total_cost'].idxmin()]
            print(f"    Best cost with high service: {best_cost_high_service['method_name']} "
                  f"(${best_cost_high_service['total_cost']:.1f})")

    print(f"\n✅ Comprehensive comparison completed!")


if __name__ == "__main__":
    results_df = run_comprehensive_comparison()