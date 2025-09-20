"""
Setup script for downloading and preparing inventory management datasets
"""

import os
import sys
from pathlib import Path
import pandas as pd
from src.data.data_sources import RetailDataLoader, InventoryDataPreprocessor, explore_dataset


def main():
    """Main setup function for data preparation"""

    print("🏪 JAX Inventory Optimizer - Data Setup")
    print("=" * 50)

    # Create data loader
    loader = RetailDataLoader()

    print("\nAvailable Dataset Options:")
    print("1. 📊 Store Item Demand Forecasting (Kaggle)")
    print("   - 50 items across 10 stores")
    print("   - 5 years of daily sales data (2013-2017)")
    print("   - Download: https://www.kaggle.com/c/demand-forecasting-kernels-only")
    print()
    print("2. 🛒 M5 Walmart Dataset (Kaggle)")
    print("   - 3,049 products across 10 stores")
    print("   - Hierarchical sales data with prices and calendar")
    print("   - Download: https://www.kaggle.com/c/m5-forecasting-accuracy")
    print()
    print("3. 🎲 Synthetic Dataset (Auto-generated)")
    print("   - Controllable demand patterns")
    print("   - Good for development and testing")

    print("\n" + "=" * 50)

    # Check if real datasets are available
    store_item_path = Path("data/train.csv")
    m5_path = Path("data/m5-forecasting")

    if store_item_path.exists():
        print("✅ Store Item Demand dataset found!")
        df = loader.load_store_item_demand()
        process_dataset(df, "Store Item Demand")

    elif m5_path.exists():
        print("✅ M5 Walmart dataset found!")
        datasets = loader.load_m5_walmart()
        if datasets:
            print("M5 dataset loaded successfully!")
            # Process M5 data here if needed

    else:
        print("⚠️  No real datasets found. Creating synthetic data...")
        df = loader.load_store_item_demand()  # Will create sample data
        process_dataset(df, "Synthetic Sample")

    print("\n🚀 Data setup complete!")
    print("\nNext steps:")
    print("1. Run: python -m src.models.traditional.eoq_baseline")
    print("2. Or start with: python experiments/compare_methods.py")


def process_dataset(df: pd.DataFrame, dataset_name: str):
    """Process and analyze a dataset"""

    print(f"\n📈 Processing {dataset_name} Dataset")
    print("-" * 40)

    # Explore dataset
    stats = explore_dataset(df)
    print(f"Dataset shape: {stats['shape']}")
    print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"Stores: {stats['num_stores']}, Items: {stats['num_items']}")
    print(f"Total sales: {stats['total_sales']:,}")
    print(f"Average daily sales: {stats['avg_daily_sales']:.2f}")
    print(f"Zero sales percentage: {stats['zero_sales_pct']:.2f}%")

    # Prepare sample SKU data
    preprocessor = InventoryDataPreprocessor()

    # Analyze multiple SKUs for variety
    sample_skus = [
        (1, 1), (1, 2), (2, 1),  # Different store-item combinations
        (3, 5), (4, 10)
    ]

    print(f"\n📊 Sample SKU Analysis:")
    print("-" * 40)

    for store_id, item_id in sample_skus:
        try:
            sku_data = preprocessor.prepare_single_sku_data(df, store_id, item_id)
            metadata = sku_data['metadata']

            print(f"Store {store_id}, Item {item_id}:")
            print(f"  Mean demand: {metadata['mean_demand']:.2f}")
            print(f"  Demand CV: {metadata['cv_demand']:.2f}")
            print(f"  Range: {metadata['min_demand']}-{metadata['max_demand']}")

        except ValueError as e:
            print(f"Store {store_id}, Item {item_id}: {e}")

    # Save processed sample for quick access
    sample_sku = preprocessor.prepare_single_sku_data(df, store_id=1, item_id=1)
    sample_df = pd.DataFrame({
        'demand': sample_sku['demand'],
        'day_of_week': sample_sku['day_of_week'],
        'month': sample_sku['month'],
        'is_promotion': sample_sku['is_promotion']
    })

    output_path = Path("data/sample_sku_data.csv")
    sample_df.to_csv(output_path, index=False)
    print(f"\n💾 Sample SKU data saved to: {output_path}")


if __name__ == "__main__":
    main()