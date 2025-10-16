"""
Basic Usage Example: Getting Started with JAX Inventory Optimizer

This example demonstrates the fundamental workflow:
1. Creating inventory states
2. Using traditional optimization methods
3. Comparing different approaches
4. Evaluating performance

Run: python examples/01_basic_usage.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.core.interfaces import InventoryState
from src.methods.traditional.eoq import EOQMethod
from src.methods.traditional.safety_stock import SafetyStockMethod
from src.methods.traditional.s_S_policy import sSPolicyMethod


def create_sample_state(inventory_level: float = 100.0) -> InventoryState:
    """Create a sample inventory state for demonstration."""
    return InventoryState(
        inventory_level=inventory_level,
        outstanding_orders=0.0,
        demand_history=[45, 52, 48, 50, 55, 47, 51, 49, 53, 46],
        time_step=0
    )


def example_1_eoq_method():
    """Example 1: Economic Order Quantity (EOQ) Method"""
    print("\n" + "="*80)
    print("Example 1: Economic Order Quantity (EOQ)")
    print("="*80)

    # Initialize EOQ method with cost parameters
    eoq = EOQMethod(
        holding_cost=1.0,      # Cost to hold one unit for one period
        ordering_cost=50.0,    # Fixed cost per order
        unit_cost=10.0         # Cost per unit
    )

    # Create inventory state
    state = create_sample_state(inventory_level=50.0)

    # Fit the model with demand history
    eoq.fit(np.array(state.demand_history))

    # Get recommendation
    action = eoq.decide(state)

    print(f"\nInventory Parameters:")
    print(f"  Current inventory level: {state.inventory_level:.2f} units")
    print(f"  Average demand: {np.mean(state.demand_history):.2f} units/period")
    print(f"  Holding cost: ${eoq.holding_cost}/unit/period")
    print(f"  Ordering cost: ${eoq.ordering_cost}/order")

    print(f"\nEOQ Recommendation:")
    print(f"  Optimal order quantity: {action.order_quantity:.2f} units")
    print(f"  Reorder point: {action.reorder_point:.2f} units")
    print(f"  Expected cost impact: ${action.expected_cost:.2f}")

    print(f"\nEOQ Formula:")
    print(f"  Q* = sqrt(2 * D * K / h)")
    print(f"  Where: D = annual demand, K = ordering cost, h = holding cost")


def example_2_safety_stock():
    """Example 2: Safety Stock Method for Service Level Management"""
    print("\n" + "="*80)
    print("Example 2: Safety Stock Method")
    print("="*80)

    # Initialize safety stock method with service level target
    safety_stock = SafetyStockMethod(
        service_level=0.95,    # 95% service level (5% stockout probability)
        holding_cost=1.0,
        ordering_cost=50.0,
        lead_time=7            # 7-day lead time
    )

    # Create inventory state
    state = create_sample_state(inventory_level=80.0)

    # Fit the model with demand history
    safety_stock.fit(np.array(state.demand_history))

    # Get recommendation
    action = safety_stock.decide(state)

    print(f"\nService Level Parameters:")
    print(f"  Target service level: {safety_stock.service_level:.1%}")
    print(f"  Lead time: {safety_stock.lead_time} periods")
    print(f"  Demand mean: {np.mean(state.demand_history):.2f}")
    print(f"  Demand std: {np.std(state.demand_history):.2f}")

    print(f"\nSafety Stock Recommendation:")
    print(f"  Order quantity: {action.order_quantity:.2f} units")
    print(f"  Safety stock level: {action.safety_stock:.2f} units")
    print(f"  Reorder point: {action.reorder_point:.2f} units")

    print(f"\nInterpretation:")
    print(f"  With 95% service level, we should maintain {action.safety_stock:.2f} units")
    print(f"  as buffer stock to protect against demand variability.")


def example_3_s_S_policy():
    """Example 3: (s,S) Policy for Batch Ordering"""
    print("\n" + "="*80)
    print("Example 3: (s,S) Policy")
    print("="*80)

    # Initialize (s,S) policy
    s_S_policy = sSPolicyMethod(
        holding_cost=1.0,
        ordering_cost=50.0,
        stockout_cost=10.0,    # Cost per unit of unmet demand
        min_order_quantity=20  # Minimum order size
    )

    # Fit the model with sample demand history
    sample_state = create_sample_state()
    s_S_policy.fit(np.array(sample_state.demand_history))

    # Test different inventory levels
    inventory_levels = [30, 50, 70, 100]

    print(f"\n(s,S) Policy Parameters:")
    print(f"  s (reorder point): {s_S_policy.s:.2f} units")
    print(f"  S (order-up-to level): {s_S_policy.S:.2f} units")
    print(f"  Policy: Order when inventory ≤ s, order up to S")

    print(f"\nOrdering Decisions at Different Inventory Levels:")
    print("-" * 60)

    for inv_level in inventory_levels:
        state = create_sample_state(inventory_level=inv_level)
        action = s_S_policy.decide(state)

        should_order = inv_level <= s_S_policy.s
        order_qty = action.order_quantity

        print(f"\n  Inventory: {inv_level:.0f} units")
        print(f"    Should order? {'YES' if should_order else 'NO'}")
        print(f"    Order quantity: {order_qty:.2f} units")
        print(f"    Final inventory: {inv_level + order_qty:.2f} units")


def example_4_method_comparison():
    """Example 4: Comparing All Traditional Methods"""
    print("\n" + "="*80)
    print("Example 4: Method Comparison")
    print("="*80)

    # Initialize all methods
    methods = {
        'EOQ': EOQMethod(holding_cost=1.0, ordering_cost=50.0, unit_cost=10.0),
        'Safety Stock': SafetyStockMethod(
            service_level=0.95, holding_cost=1.0, ordering_cost=50.0, lead_time=7
        ),
        '(s,S) Policy': sSPolicyMethod(
            holding_cost=1.0, ordering_cost=50.0, stockout_cost=10.0
        )
    }

    # Test state
    state = create_sample_state(inventory_level=60.0)

    # Fit all methods
    for method in methods.values():
        method.fit(np.array(state.demand_history))

    print(f"\nCurrent State:")
    print(f"  Inventory: {state.inventory_level:.2f} units")
    print(f"  Average demand: {np.mean(state.demand_history):.2f} units/period")

    print(f"\nMethod Recommendations:")
    print("-" * 60)

    for method_name, method in methods.items():
        action = method.decide(state)
        print(f"\n{method_name}:")
        print(f"  Order quantity: {action.order_quantity:.2f} units")
        if action.expected_cost is not None:
            print(f"  Expected cost: ${action.expected_cost:.2f}")
        if hasattr(action, 'safety_stock') and action.safety_stock is not None:
            print(f"  Safety stock: {action.safety_stock:.2f} units")


def example_5_simulation():
    """Example 5: Simple Simulation Over Time"""
    print("\n" + "="*80)
    print("Example 5: Time Series Simulation")
    print("="*80)

    # Initialize method
    eoq = EOQMethod(holding_cost=1.0, ordering_cost=50.0, unit_cost=10.0)

    # Simulation parameters
    num_periods = 20
    initial_inventory = 100.0
    demand_mean = 50.0
    demand_std = 10.0

    # Fit the model with initial demand history
    initial_demand = np.random.normal(demand_mean, demand_std, 30)
    eoq.fit(initial_demand)

    # Simulate
    inventory_levels = [initial_inventory]
    order_quantities = []
    total_cost = 0.0

    print(f"\nSimulation Setup:")
    print(f"  Periods: {num_periods}")
    print(f"  Initial inventory: {initial_inventory:.0f} units")
    print(f"  Demand: N({demand_mean:.0f}, {demand_std:.0f})")

    print(f"\nSimulation Results:")
    print("-" * 80)
    print(f"{'Period':<8} {'Demand':<10} {'Order':<10} {'Inventory':<12} {'Cost':<10}")
    print("-" * 80)

    for t in range(num_periods):
        # Generate random demand
        demand = max(0, np.random.normal(demand_mean, demand_std))

        # Create state
        state = InventoryState(
            inventory_level=inventory_levels[-1],
            outstanding_orders=0.0,
            demand_history=[demand_mean] * 10,  # Simplified
            time_step=t
        )

        # Get order decision
        action = eoq.decide(state)
        order_qty = action.order_quantity

        # Update inventory
        new_inventory = max(0, inventory_levels[-1] + order_qty - demand)

        # Calculate period cost
        holding_cost = inventory_levels[-1] * eoq.holding_cost
        ordering_cost = eoq.ordering_cost if order_qty > 0 else 0
        period_cost = holding_cost + ordering_cost
        total_cost += period_cost

        # Store results
        inventory_levels.append(new_inventory)
        order_quantities.append(order_qty)

        # Print period summary
        print(f"{t:<8} {demand:<10.1f} {order_qty:<10.1f} {new_inventory:<12.1f} ${period_cost:<9.2f}")

    print("-" * 80)
    print(f"\nSimulation Summary:")
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Average inventory: {np.mean(inventory_levels):.2f} units")
    print(f"  Number of orders: {sum(1 for q in order_quantities if q > 0)}")
    print(f"  Average order size: {np.mean([q for q in order_quantities if q > 0]):.2f} units")
    print(f"  Stockout periods: {sum(1 for inv in inventory_levels if inv < 1)}")


def main():
    """Run all basic examples."""
    print("\n" + "="*80)
    print("JAX INVENTORY OPTIMIZER - BASIC USAGE EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates the fundamental features of the optimizer.")
    print("Each example showcases a different traditional optimization method.")

    # Run examples
    example_1_eoq_method()
    example_2_safety_stock()
    example_3_s_S_policy()
    example_4_method_comparison()
    example_5_simulation()

    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Run advanced examples: python examples/02_advanced_features.py")
    print("  2. Explore ML methods: python examples/03_ml_forecasting.py")
    print("  3. Try cost optimization: python examples/04_cost_optimization.py")
    print("  4. Check API examples: python examples/05_api_client.py")
    print("\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run all examples
    main()
