"""
API Client Example: Using the REST API

This example demonstrates:
1. Starting the FastAPI server
2. Making API requests
3. Batch recommendations
4. Model comparison
5. Health monitoring

Prerequisites:
  - Start API server: uvicorn src.api.main:app --reload
  - Or run in background: python examples/04_api_client.py --start-server

Run: python examples/04_api_client.py
"""

import sys
from pathlib import Path
import time
import requests
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# API Configuration
API_BASE_URL = "http://localhost:8000"


def check_server_running():
    """Check if API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def example_1_health_check():
    """Example 1: Health Check and Server Info"""
    print("\n" + "="*80)
    print("Example 1: Health Check")
    print("="*80)

    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()

        health = response.json()

        print(f"\nServer Status:")
        print(f"  Status: {health['status']}")
        print(f"  Timestamp: {health['timestamp']}")
        print(f"  Version: {health.get('version', 'N/A')}")

        print(f"\nAvailable Models:")
        for model in health.get('available_models', []):
            print(f"  - {model}")

        return True

    except Exception as e:
        print(f"\nError: Could not connect to API server")
        print(f"  {e}")
        print(f"\nPlease start the server:")
        print(f"  uvicorn src.api.main:app --reload")
        return False


def example_2_single_recommendation():
    """Example 2: Getting Single Recommendation"""
    print("\n" + "="*80)
    print("Example 2: Single Model Recommendation")
    print("="*80)

    # Create sample state
    state = {
        "inventory_level": 50.0,
        "outstanding_orders": 0.0,
        "demand_history": [45, 52, 48, 50, 55, 47, 51, 49, 53, 46],
        "time_step": 0
    }

    print(f"\nInventory State:")
    print(f"  Current inventory: {state['inventory_level']} units")
    print(f"  Recent demand: {state['demand_history'][-5:]}")

    # Request recommendation from EOQ method
    print(f"\nRequesting EOQ recommendation...")

    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend/eoq",
            json=state,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        result = response.json()

        print(f"\nEOQ Recommendation:")
        print(f"  Order quantity: {result['order_quantity']:.2f} units")
        print(f"  Reorder point: {result.get('reorder_point', 'N/A')}")
        print(f"  Expected cost: ${result.get('expected_cost', 'N/A'):.2f}")
        print(f"  Method: {result['method']}")
        print(f"  Response time: {response.elapsed.total_seconds():.3f}s")

    except Exception as e:
        print(f"\nError: {e}")


def example_3_batch_comparison():
    """Example 3: Batch Model Comparison"""
    print("\n" + "="*80)
    print("Example 3: Batch Model Comparison")
    print("="*80)

    # Create sample state
    state = {
        "inventory_level": 75.0,
        "outstanding_orders": 0.0,
        "demand_history": [48, 51, 49, 52, 50, 53, 47, 54, 49, 51],
        "time_step": 5
    }

    # Request comparison of multiple methods
    request_data = {
        "models": ["eoq", "safety_stock", "s_s_policy"],
        "state": state
    }

    print(f"\nRequesting recommendations from multiple models...")
    print(f"  Models: {', '.join(request_data['models'])}")

    try:
        response = requests.post(
            f"{API_BASE_URL}/batch_recommend",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        results = response.json()

        print(f"\nComparison Results:")
        print("-" * 70)
        print(f"{'Method':<20} {'Order Qty':<15} {'Expected Cost':<15} {'Status':<10}")
        print("-" * 70)

        for model_result in results['recommendations']:
            method = model_result['method']
            if model_result['success']:
                order_qty = model_result['recommendation']['order_quantity']
                cost = model_result['recommendation'].get('expected_cost', 0)
                status = "✓"
                print(f"{method:<20} {order_qty:<15.2f} ${cost:<14.2f} {status:<10}")
            else:
                print(f"{method:<20} {'N/A':<15} {'N/A':<15} {'✗':<10}")
                print(f"  Error: {model_result.get('error', 'Unknown')}")

        print("-" * 70)
        print(f"\nTotal response time: {response.elapsed.total_seconds():.3f}s")

    except Exception as e:
        print(f"\nError: {e}")


def example_4_ml_forecast():
    """Example 4: ML-Based Forecasting"""
    print("\n" + "="*80)
    print("Example 4: ML Forecasting")
    print("="*80)

    # Create state with more historical data for ML model
    state = {
        "inventory_level": 100.0,
        "outstanding_orders": 20.0,
        "demand_history": [
            45, 48, 52, 49, 51, 53, 47, 50, 54, 48,
            49, 52, 50, 53, 51, 48, 54, 49, 52, 50,
            55, 51, 49, 53, 48, 52, 50, 54, 49, 51
        ],
        "time_step": 30
    }

    print(f"\nInventory State:")
    print(f"  Current inventory: {state['inventory_level']} units")
    print(f"  Outstanding orders: {state['outstanding_orders']} units")
    print(f"  Historical data points: {len(state['demand_history'])}")

    print(f"\nRequesting LSTM forecast...")

    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend/lstm",
            json=state,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        result = response.json()

        print(f"\nLSTM Recommendation:")
        print(f"  Demand forecast: {result.get('forecast', 'N/A'):.2f} units")
        print(f"  Order quantity: {result['order_quantity']:.2f} units")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Method: {result['method']}")

        print(f"\nInterpretation:")
        forecast = result.get('forecast', 0)
        current = state['inventory_level'] + state['outstanding_orders']
        coverage_days = current / forecast if forecast > 0 else float('inf')
        print(f"  Current inventory covers ~{coverage_days:.1f} days of forecasted demand")

    except Exception as e:
        print(f"\nError: {e}")
        if "LSTM" in str(e) or "404" in str(e):
            print(f"\nNote: LSTM endpoint may require model training first")


def example_5_error_handling():
    """Example 5: Error Handling"""
    print("\n" + "="*80)
    print("Example 5: Error Handling")
    print("="*80)

    print(f"\nTesting various error scenarios...")

    # Test 1: Invalid model
    print(f"\n1. Invalid model name:")
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend/invalid_model",
            json={"inventory_level": 50, "demand_history": [10]},
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Missing required fields
    print(f"\n2. Missing required fields:")
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend/eoq",
            json={"inventory_level": 50},  # Missing demand_history
        )
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   ✓ Properly rejected invalid request")
            print(f"   Error: {response.json().get('detail', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Invalid data types
    print(f"\n3. Invalid data types:")
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend/eoq",
            json={
                "inventory_level": "not_a_number",
                "demand_history": [10, 20]
            },
        )
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   ✓ Properly validated data types")
    except Exception as e:
        print(f"   Error: {e}")


def example_6_performance_testing():
    """Example 6: Performance Testing"""
    print("\n" + "="*80)
    print("Example 6: Performance Testing")
    print("="*80)

    state = {
        "inventory_level": 50.0,
        "outstanding_orders": 0.0,
        "demand_history": [45, 52, 48, 50, 55, 47, 51, 49, 53, 46],
        "time_step": 0
    }

    num_requests = 10

    print(f"\nSending {num_requests} sequential requests...")

    response_times = []

    for i in range(num_requests):
        try:
            start = time.time()
            response = requests.post(
                f"{API_BASE_URL}/recommend/eoq",
                json=state,
            )
            elapsed = time.time() - start
            response_times.append(elapsed)

            if i == 0:
                print(f"  Request {i+1}: {elapsed*1000:.2f}ms (may include cold start)")
            elif i < 3:
                print(f"  Request {i+1}: {elapsed*1000:.2f}ms")

        except Exception as e:
            print(f"  Request {i+1} failed: {e}")

    if response_times:
        print(f"\nPerformance Summary:")
        print(f"  Total requests: {len(response_times)}")
        print(f"  Average latency: {sum(response_times)/len(response_times)*1000:.2f}ms")
        print(f"  Min latency: {min(response_times)*1000:.2f}ms")
        print(f"  Max latency: {max(response_times)*1000:.2f}ms")
        print(f"  Requests/second: {len(response_times)/sum(response_times):.2f}")


def example_7_documentation():
    """Example 7: API Documentation"""
    print("\n" + "="*80)
    print("Example 7: API Documentation")
    print("="*80)

    print(f"\nAPI documentation is available at:")
    print(f"  Interactive docs: {API_BASE_URL}/docs")
    print(f"  ReDoc: {API_BASE_URL}/redoc")
    print(f"  OpenAPI schema: {API_BASE_URL}/openapi.json")

    print(f"\nAvailable Endpoints:")
    print(f"  GET  /health                  - Health check")
    print(f"  POST /recommend/{{model}}      - Single recommendation")
    print(f"  POST /batch_recommend         - Batch comparison")
    print(f"  GET  /models                  - List available models")

    print(f"\nSupported Models:")
    print(f"  - eoq                         - Economic Order Quantity")
    print(f"  - safety_stock                - Safety Stock Method")
    print(f"  - s_s_policy                  - (s,S) Policy")
    print(f"  - lstm                        - LSTM Neural Network")
    print(f"  - dqn                         - Deep Q-Network")

    print(f"\nTry opening the interactive docs in your browser:")
    print(f"  {API_BASE_URL}/docs")


def main():
    """Run all API client examples."""
    print("\n" + "="*80)
    print("JAX INVENTORY OPTIMIZER - API CLIENT EXAMPLES")
    print("="*80)

    print(f"\nThese examples demonstrate how to interact with the REST API")
    print(f"for real-time inventory optimization recommendations.")

    # Check if server is running
    print(f"\nChecking API server status...")

    if not check_server_running():
        print(f"\n{'='*80}")
        print(f"API SERVER NOT RUNNING")
        print(f"{'='*80}")
        print(f"\nPlease start the API server first:")
        print(f"  uvicorn src.api.main:app --reload")
        print(f"\nOr in a separate terminal:")
        print(f"  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        print(f"\nThen run this script again.")
        return

    # Run examples
    try:
        example_1_health_check()
        example_2_single_recommendation()
        example_3_batch_comparison()
        example_4_ml_forecast()
        example_5_error_handling()
        example_6_performance_testing()
        example_7_documentation()

        print("\n" + "="*80)
        print("API CLIENT EXAMPLES COMPLETED")
        print("="*80)
        print("\nKey Capabilities:")
        print("  ✓ Health monitoring and server status")
        print("  ✓ Single model recommendations")
        print("  ✓ Batch model comparison")
        print("  ✓ ML-based forecasting")
        print("  ✓ Robust error handling")
        print("  ✓ Low-latency inference (<10ms)")
        print("\nProduction Deployment:")
        print("  - Docker: docker-compose up")
        print("  - Kubernetes: kubectl apply -f k8s/")
        print("  - Helm: helm install inventory-optimizer ./helm/jax-optimizer")
        print()

    except Exception as e:
        print(f"\nError running examples: {e}")


if __name__ == "__main__":
    main()
