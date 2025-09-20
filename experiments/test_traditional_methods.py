"""
Comprehensive test of all traditional inventory methods

This script tests and compares all implemented traditional methods
using the sample retail data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from methods.traditional.eoq import EOQMethod, EOQWithBackorders
from methods.traditional.safety_stock import SafetyStockMethod, AdaptiveSafetyStock
from methods.traditional.s_S_policy import sSPolicyMethod, AdaptivesSPolicy
from core.interfaces import InventoryState, InventoryEvaluator


def load_sample_data():
    """Load sample retail data"""
    data_path = Path(__file__).parent.parent / 'data' / 'sample_retail_data.csv'

    if not data_path.exists():
        print("Sample data not found. Please run create_sample_data.py first.")
        return None

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def prepare_single_sku_data(df, store_id=1, item_id=1):
    """Extract single SKU data for testing"""
    sku_data = df[(df['store'] == store_id) & (df['item'] == item_id)].copy()
    sku_data = sku_data.sort_values('date')

    # Split into train/test
    n = len(sku_data)
    train_size = int(0.8 * n)

    train_data = sku_data.iloc[:train_size]['sales'].values
    test_data = sku_data.iloc[train_size:]['sales'].values

    return train_data, test_data


def test_traditional_methods():
    """Test all traditional methods"""

    print("🏪 Traditional Inventory Methods - Comprehensive Test")
    print("=" * 60)

    # Load data
    df = load_sample_data()
    if df is None:
        return

    train_demand, test_demand = prepare_single_sku_data(df)

    print(f"📊 Data Summary:")
    print(f"  Train periods: {len(train_demand)}")
    print(f"  Test periods: {len(test_demand)}")
    print(f"  Train mean demand: {np.mean(train_demand):.2f}")
    print(f"  Train std demand: {np.std(train_demand):.2f}")

    # Initialize methods
    methods = {
        'EOQ': EOQMethod(holding_cost=2.0, ordering_cost=50.0),
        'EOQ_Backorders': EOQWithBackorders(holding_cost=2.0, ordering_cost=50.0, backorder_cost=5.0),
        'SafetyStock_Normal': SafetyStockMethod(method='normal'),
        'SafetyStock_Empirical': SafetyStockMethod(method='empirical'),
        'AdaptiveSafetyStock': AdaptiveSafetyStock(window_size=30),
        'sS_Policy_Analytical': sSPolicyMethod(optimization_method='analytical'),
        'sS_Policy_Numerical': sSPolicyMethod(optimization_method='numerical'),
        'Adaptive_sS_Policy': AdaptivesSPolicy(target_service_level=0.95)
    }

    # Fit all methods
    print(f"\n🔧 Fitting Methods...")
    fitted_methods = {}

    for name, method in methods.items():
        try:
            method.fit(train_demand)
            fitted_methods[name] = method
            print(f"  ✅ {name}: Fitted successfully")
        except Exception as e:
            print(f"  ❌ {name}: Failed to fit - {e}")

    # Test recommendations
    print(f"\n📋 Testing Recommendations:")
    print("-" * 60)

    # Create test state
    current_state = InventoryState(
        inventory_level=50.0,
        outstanding_orders=20.0,
        demand_history=train_demand[-30:],
        time_step=len(train_demand)
    )

    results = {}

    for name, method in fitted_methods.items():
        try:
            # Get recommendation
            action = method.recommend_action(current_state)

            # Get demand prediction
            demand_pred = method.predict_demand(current_state, horizon=7)

            # Get policy description
            if hasattr(method, 'get_policy_description'):
                policy_desc = method.get_policy_description()
            else:
                policy_desc = f"{name} policy"

            results[name] = {
                'order_quantity': action.order_quantity,
                'demand_prediction': np.mean(demand_pred),
                'policy_description': policy_desc,
                'parameters': method.get_parameters()
            }

            print(f"{name:25} | Order: {action.order_quantity:6.1f} | "
                  f"Pred Demand: {np.mean(demand_pred):6.1f}")

        except Exception as e:
            print(f"{name:25} | Error: {e}")
            results[name] = {'error': str(e)}

    # Compare key parameters
    print(f"\n🔍 Method Parameters Comparison:")
    print("-" * 60)

    comparison_metrics = [
        ('EOQ', 'optimal_order_quantity'),
        ('SafetyStock_Normal', 'safety_stock'),
        ('SafetyStock_Empirical', 'safety_stock'),
        ('sS_Policy_Analytical', 's_reorder_point'),
        ('sS_Policy_Analytical', 'S_order_up_to')
    ]

    for method_name, param_name in comparison_metrics:
        if method_name in results and 'parameters' in results[method_name]:
            params = results[method_name]['parameters']
            if param_name in params:
                print(f"{method_name:25} | {param_name:20}: {params[param_name]:8.2f}")

    # Simulate performance
    print(f"\n🎯 Performance Simulation:")
    print("-" * 60)

    evaluator = InventoryEvaluator()

    # Create test states for evaluation
    test_states = []
    for i in range(len(test_demand)):
        state = InventoryState(
            inventory_level=50.0,  # Starting inventory
            outstanding_orders=0.0,
            demand_history=train_demand[-30:],  # Use recent history
            time_step=len(train_demand) + i
        )
        test_states.append(state)

    performance_results = {}

    for name, method in fitted_methods.items():
        if name in results and 'error' not in results[name]:
            try:
                result = evaluator.evaluate_method(method, test_states[:10], test_demand[:10])
                performance_results[name] = {
                    'total_cost': result.total_cost,
                    'service_level': result.service_level,
                    'forecast_accuracy': result.forecast_accuracy
                }

                print(f"{name:25} | Cost: {result.total_cost:6.1f} | "
                      f"Service: {result.service_level:5.1%} | "
                      f"Accuracy: {result.forecast_accuracy:5.1%}")

            except Exception as e:
                print(f"{name:25} | Evaluation error: {e}")

    # Summary
    print(f"\n📈 Summary:")
    print("-" * 60)

    if performance_results:
        # Best by cost
        best_cost = min(performance_results.items(), key=lambda x: x[1]['total_cost'])
        print(f"🏆 Best Cost: {best_cost[0]} (${best_cost[1]['total_cost']:.1f})")

        # Best by service level
        best_service = max(performance_results.items(), key=lambda x: x[1]['service_level'])
        print(f"🎯 Best Service: {best_service[0]} ({best_service[1]['service_level']:.1%})")

        # Best forecast accuracy
        best_accuracy = max(performance_results.items(), key=lambda x: x[1]['forecast_accuracy'])
        print(f"🎪 Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['forecast_accuracy']:.1%})")

    print(f"\n✅ Traditional Methods Test Completed!")

    return results, performance_results


if __name__ == "__main__":
    results, performance = test_traditional_methods()