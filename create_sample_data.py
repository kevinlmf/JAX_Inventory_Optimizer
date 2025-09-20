"""
Simple script to create sample retail inventory data without JAX dependencies
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_sample_retail_data():
    """Create realistic sample retail data for inventory optimization"""

    print("🏪 Creating Sample Retail Dataset...")
    print("=" * 50)

    np.random.seed(42)

    # Parameters
    num_stores = 5
    num_items = 10
    start_date = '2021-01-01'
    end_date = '2023-12-31'

    # Create date range
    dates = pd.date_range(start_date, end_date, freq='D')
    print(f"📅 Date range: {start_date} to {end_date} ({len(dates)} days)")

    data = []

    for store in range(1, num_stores + 1):
        print(f"🏬 Processing Store {store}...")

        for item in range(1, num_items + 1):
            # Each store-item has different characteristics
            base_demand = np.random.uniform(15, 80)  # Base daily demand
            seasonality_strength = np.random.uniform(0.2, 0.5)
            trend_rate = np.random.uniform(-0.001, 0.002)  # Small trend

            for i, date in enumerate(dates):
                # Seasonal pattern (annual cycle)
                day_of_year = date.dayofyear
                seasonal = seasonality_strength * np.sin(2 * np.pi * day_of_year / 365.25)

                # Weekly pattern (lower demand on weekends)
                weekday_effect = -0.3 if date.weekday() >= 5 else 0.1

                # Trend component
                trend = trend_rate * i

                # Special events/promotions (random spikes)
                promotion_prob = 0.03  # 3% chance
                promotion_effect = 0
                if np.random.random() < promotion_prob:
                    promotion_effect = np.random.uniform(1.5, 3.0)

                # Holiday effects
                holiday_effect = 0
                if date.month == 12:  # December boost
                    holiday_effect = 0.4
                elif date.month in [6, 7]:  # Summer boost
                    holiday_effect = 0.2

                # Random noise
                noise = np.random.normal(0, 0.15)

                # Calculate final demand
                demand_multiplier = (1 + seasonal + weekday_effect + trend +
                                   promotion_effect + holiday_effect + noise)
                daily_demand = max(0, base_demand * demand_multiplier)

                # Round to integers (unit sales)
                sales = max(0, int(np.round(daily_demand)))

                data.append({
                    'date': date,
                    'store': store,
                    'item': item,
                    'sales': sales
                })

    # Create DataFrame
    df = pd.DataFrame(data)
    print(f"\n📊 Dataset created: {df.shape[0]} records")

    # Basic statistics
    print("\n📈 Dataset Statistics:")
    print(f"  Stores: {df['store'].nunique()}")
    print(f"  Items: {df['item'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total sales: {df['sales'].sum():,}")
    print(f"  Average daily sales: {df['sales'].mean():.2f}")
    print(f"  Sales std: {df['sales'].std():.2f}")

    # Save to file
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / "sample_retail_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Data saved to: {output_file}")

    # Analyze a sample SKU
    sample_sku = df[(df['store'] == 1) & (df['item'] == 1)].copy()
    sample_sku['day_of_week'] = sample_sku['date'].dt.day_name()

    print(f"\n🔍 Sample SKU Analysis (Store 1, Item 1):")
    print(f"  Mean daily sales: {sample_sku['sales'].mean():.2f}")
    print(f"  Sales std: {sample_sku['sales'].std():.2f}")
    print(f"  Min/Max sales: {sample_sku['sales'].min()}/{sample_sku['sales'].max()}")

    # Day of week analysis
    dow_stats = sample_sku.groupby('day_of_week')['sales'].mean().round(2)
    print(f"  Day of week patterns:")
    for day, avg_sales in dow_stats.items():
        print(f"    {day}: {avg_sales}")

    # Save sample SKU for quick testing
    sample_file = data_dir / "sample_sku_data.csv"
    sample_sku[['date', 'sales']].to_csv(sample_file, index=False)
    print(f"\n💾 Sample SKU data saved to: {sample_file}")

    return df


def analyze_demand_patterns(df):
    """Analyze demand patterns in the dataset"""

    print("\n🔬 Demand Pattern Analysis:")
    print("=" * 50)

    # Overall patterns
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    # Monthly seasonality
    monthly_avg = df.groupby('month')['sales'].mean()
    print(f"\n📅 Monthly Average Sales:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i, month in enumerate(months, 1):
        print(f"  {month}: {monthly_avg[i]:.2f}")

    # Day of week patterns
    dow_avg = df.groupby('day_of_week')['sales'].mean()
    print(f"\n📊 Day of Week Average Sales:")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, day in enumerate(days):
        print(f"  {day}: {dow_avg[i]:.2f}")

    # Store and item variations
    store_avg = df.groupby('store')['sales'].mean()
    item_avg = df.groupby('item')['sales'].mean()

    print(f"\n🏬 Store Performance:")
    for store in sorted(store_avg.index):
        print(f"  Store {store}: {store_avg[store]:.2f}")

    print(f"\n🛍️ Item Performance (Top 5):")
    top_items = item_avg.nlargest(5)
    for item, avg_sales in top_items.items():
        print(f"  Item {item}: {avg_sales:.2f}")

    return df


if __name__ == "__main__":
    # Create sample data
    df = create_sample_retail_data()

    # Analyze patterns
    df = analyze_demand_patterns(df)

    print("\n✅ Sample data creation complete!")
    print("\nNext steps:")
    print("1. Explore the data: data/sample_retail_data.csv")
    print("2. Run traditional models on the sample SKU data")
    print("3. Implement ML/RL approaches for comparison")