"""
Data Sources for Inventory Optimization

This module provides access to various retail datasets suitable for
inventory management and demand forecasting research.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import requests
import os
from pathlib import Path


class DatasetConfig:
    """Configuration for different datasets"""

    STORE_ITEM_DEMAND = {
        'name': 'Store Item Demand Forecasting',
        'kaggle_competition': 'demand-forecasting-kernels-only',
        'description': 'Daily sales data for 50 items across 10 stores (2013-2017)',
        'features': ['date', 'store', 'item', 'sales'],
        'time_span': '5 years',
        'frequency': 'daily',
        'num_stores': 10,
        'num_items': 50
    }

    M5_WALMART = {
        'name': 'M5 Forecasting - Walmart',
        'kaggle_competition': 'm5-forecasting-accuracy',
        'description': 'Hierarchical sales data for 3,049 products across 10 stores',
        'features': ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'sales'],
        'time_span': '5+ years (2011-2016)',
        'frequency': 'daily',
        'num_stores': 10,
        'num_items': 3049,
        'files': ['sales_train.csv', 'calendar.csv', 'sell_prices.csv']
    }

    SYNTHETIC_RETAIL = {
        'name': 'Synthetic Retail Dataset',
        'description': 'Generated data with controllable demand patterns',
        'features': ['date', 'demand', 'seasonality', 'trend', 'promotions'],
        'controllable': True
    }


class RetailDataLoader:
    """Load and preprocess retail datasets for inventory optimization"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def load_store_item_demand(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load Store Item Demand Forecasting dataset

        Expected format:
        - date: Date of sale
        - store: Store identifier (1-10)
        - item: Item identifier (1-50)
        - sales: Number of units sold
        """
        if file_path is None:
            file_path = self.data_dir / "train.csv"

        if not file_path.exists():
            print(f"Dataset not found at {file_path}")
            print("Please download from: https://www.kaggle.com/c/demand-forecasting-kernels-only")
            return self._create_sample_data()

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def load_m5_walmart(self, data_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load M5 Walmart dataset

        Returns dictionary with:
        - sales_train: Main sales data
        - calendar: Date features
        - prices: Price information
        """
        if data_dir is None:
            data_dir = self.data_dir / "m5-forecasting"

        files = {
            'sales_train': data_dir / 'sales_train_validation.csv',
            'calendar': data_dir / 'calendar.csv',
            'prices': data_dir / 'sell_prices.csv'
        }

        datasets = {}
        for name, file_path in files.items():
            if file_path.exists():
                datasets[name] = pd.read_csv(file_path)
            else:
                print(f"File not found: {file_path}")
                print("Please download M5 dataset from: https://www.kaggle.com/c/m5-forecasting-accuracy")

        return datasets

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for development/testing"""
        print("Creating sample synthetic dataset for development...")

        np.random.seed(42)

        # Date range: 3 years of daily data
        dates = pd.date_range('2021-01-01', '2023-12-31', freq='D')
        stores = range(1, 6)  # 5 stores
        items = range(1, 21)   # 20 items

        data = []

        for store in stores:
            for item in items:
                # Base demand varies by store and item
                base_demand = np.random.uniform(10, 100)

                for date in dates:
                    # Seasonal pattern (annual cycle)
                    day_of_year = date.dayofyear
                    seasonal = 0.3 * np.sin(2 * np.pi * day_of_year / 365.25)

                    # Weekly pattern (lower on weekends)
                    weekly = -0.2 if date.weekday() >= 5 else 0

                    # Random promotions (5% chance)
                    promotion = 1.5 if np.random.random() < 0.05 else 0

                    # Noise
                    noise = np.random.normal(0, 0.1)

                    # Calculate demand
                    demand_factor = 1 + seasonal + weekly + promotion + noise
                    sales = max(0, int(base_demand * demand_factor))

                    data.append({
                        'date': date,
                        'store': store,
                        'item': item,
                        'sales': sales
                    })

        df = pd.DataFrame(data)

        # Save sample data
        sample_file = self.data_dir / "sample_retail_data.csv"
        df.to_csv(sample_file, index=False)
        print(f"Sample data saved to: {sample_file}")

        return df


class InventoryDataPreprocessor:
    """Preprocess retail data for inventory optimization"""

    def __init__(self):
        pass

    def prepare_single_sku_data(
        self,
        df: pd.DataFrame,
        store_id: int,
        item_id: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract single SKU time series for inventory optimization

        Args:
            df: Full retail dataset
            store_id: Target store
            item_id: Target item

        Returns:
            Dictionary with demand time series and features
        """
        # Filter for specific store-item combination
        mask = (df['store'] == store_id) & (df['item'] == item_id)
        sku_data = df[mask].sort_values('date').copy()

        if len(sku_data) == 0:
            raise ValueError(f"No data found for store {store_id}, item {item_id}")

        # Extract demand time series
        demand_series = sku_data['sales'].values
        dates = sku_data['date'].values

        # Create time-based features
        sku_data['day_of_week'] = sku_data['date'].dt.dayofweek
        sku_data['day_of_year'] = sku_data['date'].dt.dayofyear
        sku_data['month'] = sku_data['date'].dt.month
        sku_data['quarter'] = sku_data['date'].dt.quarter

        # Rolling statistics for demand patterns
        sku_data['demand_ma_7'] = sku_data['sales'].rolling(7, min_periods=1).mean()
        sku_data['demand_ma_30'] = sku_data['sales'].rolling(30, min_periods=1).mean()
        sku_data['demand_std_7'] = sku_data['sales'].rolling(7, min_periods=1).std()

        # Detect promotions (unusually high sales)
        threshold = sku_data['sales'].quantile(0.9)
        sku_data['is_promotion'] = (sku_data['sales'] > threshold).astype(int)

        return {
            'demand': demand_series,
            'dates': dates,
            'day_of_week': sku_data['day_of_week'].values,
            'day_of_year': sku_data['day_of_year'].values,
            'month': sku_data['month'].values,
            'quarter': sku_data['quarter'].values,
            'demand_ma_7': sku_data['demand_ma_7'].values,
            'demand_ma_30': sku_data['demand_ma_30'].values,
            'demand_std_7': sku_data['demand_std_7'].fillna(0).values,
            'is_promotion': sku_data['is_promotion'].values,
            'metadata': {
                'store_id': store_id,
                'item_id': item_id,
                'mean_demand': np.mean(demand_series),
                'std_demand': np.std(demand_series),
                'cv_demand': np.std(demand_series) / np.mean(demand_series),
                'min_demand': np.min(demand_series),
                'max_demand': np.max(demand_series)
            }
        }

    def create_train_test_split(
        self,
        data: Dict[str, np.ndarray],
        test_size: float = 0.2
    ) -> Tuple[Dict, Dict]:
        """Split data into train/test sets"""
        n = len(data['demand'])
        split_point = int(n * (1 - test_size))

        train_data = {k: v[:split_point] if isinstance(v, np.ndarray) else v
                     for k, v in data.items()}
        test_data = {k: v[split_point:] if isinstance(v, np.ndarray) else v
                    for k, v in data.items()}

        return train_data, test_data


# Usage example and data exploration utilities
def explore_dataset(df: pd.DataFrame) -> Dict:
    """Generate dataset statistics and insights"""

    stats = {
        'shape': df.shape,
        'date_range': (df['date'].min(), df['date'].max()),
        'num_stores': df['store'].nunique(),
        'num_items': df['item'].nunique(),
        'total_sales': df['sales'].sum(),
        'avg_daily_sales': df['sales'].mean(),
        'sales_distribution': df['sales'].describe(),
        'zero_sales_pct': (df['sales'] == 0).mean() * 100
    }

    return stats


def generate_demand_data(
    n_samples: int = 1000,
    mean: float = 50.0,
    std: float = 10.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic demand data for testing and demos.

    Args:
        n_samples: Number of demand observations to generate
        mean: Mean demand level
        std: Standard deviation of demand
        seed: Random seed for reproducibility

    Returns:
        Array of demand values
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate base demand with normal distribution
    demand = np.random.normal(mean, std, n_samples)

    # Add seasonality
    t = np.arange(n_samples)
    seasonal = 0.2 * mean * np.sin(2 * np.pi * t / 52)  # Weekly seasonality

    # Combine and ensure non-negative
    demand = demand + seasonal
    demand = np.maximum(demand, 0)

    return demand


if __name__ == "__main__":
    # Example usage
    loader = RetailDataLoader()

    # Load data (will create sample if real data not available)
    df = loader.load_store_item_demand()

    # Explore dataset
    stats = explore_dataset(df)
    print("Dataset Statistics:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Preprocess for single SKU
    preprocessor = InventoryDataPreprocessor()
    sku_data = preprocessor.prepare_single_sku_data(df, store_id=1, item_id=1)

    print(f"\nSingle SKU Data Shape: {len(sku_data['demand'])}")
    print(f"Mean demand: {sku_data['metadata']['mean_demand']:.2f}")
    print(f"Demand CV: {sku_data['metadata']['cv_demand']:.2f}")