import pandas as pd
import numpy as np

# Load clean orders and customers
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

# Keep only completed orders for EDA
completed_orders = orders[orders['status'] == 'completed'].copy()

# Merge for Q2 Cross-check
merged = completed_orders.merge(customers[['customer_id', 'city']], on='customer_id', how='left')
merged = merged.merge(products[['product_id', 'category']], on='product_id', how='left')

print("=== Pandas Cross-Check (Q2) ===")
pivot = pd.pivot_table(merged, values='order_amount', index='city', columns='category', aggfunc='sum', margins=True)
print(f"Grand Total from Pivot: {pivot.loc['All', 'All']:.2f}")

print("\n=== Groupby Named Aggregation ===")
city_stats = merged.groupby('city').agg(
    sum_amount=('order_amount', 'sum'),
    mean_amount=('order_amount', 'mean'),
    count_orders=('order_amount', 'count')
)
print(city_stats.loc[['Bengaluru', 'Mumbai']])

print("\n=== Order Amount Statistics ===")
amt = completed_orders['order_amount']
q1 = amt.quantile(0.25)
q3 = amt.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers_above = (amt > upper_fence).sum()
outliers_below = (amt < lower_fence).sum()

print(f"Mean: {amt.mean():.2f}")
print(f"Median: {amt.median():.2f}")
print(f"Skewness: {amt.skew():.2f}")
print(f"Q1: {q1:.2f}, Q3: {q3:.2f}")
print(f"Lower Fence: {lower_fence:.2f}, Upper Fence: {upper_fence:.2f}")
print(f"Outliers Above: {outliers_above}, Outliers Below: {outliers_below}")

print("\n=== Correlation Matrix ===")
corr_cols = ['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']
corr_matrix = completed_orders[corr_cols].corr()
print(corr_matrix)