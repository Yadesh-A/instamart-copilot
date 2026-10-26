import pandas as pd
import numpy as np
import sqlite3
import os

# Create paths
RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
DB_PATH = "data/instamart.db"

# --- 1. LOAD DATA ---
customers = pd.read_csv(f"{RAW_DIR}/customers.csv")
products = pd.read_csv(f"{RAW_DIR}/products.csv")
orders = pd.read_csv(f"{RAW_DIR}/orders.csv")
tickets = pd.read_csv(f"{RAW_DIR}/tickets.csv", keep_default_na=False)

# --- 2. CLEAN CUSTOMERS ---
# a. Standardize city
customers['city'] = customers['city'].str.strip().str.title()
# b. Standardize gender
def map_gender(g):
    if pd.isna(g) or g == "": return "unknown"
    g_lower = str(g).lower().strip()
    if g_lower in ["male", "m"]: return "male"
    if g_lower in ["female", "f"]: return "female"
    return "unknown"
customers['gender'] = customers['gender'].apply(map_gender)
# c. Handle Age
customers['age'] = pd.to_numeric(customers['age'], errors='coerce')
customers.loc[(customers['age'] < 10) | (customers['age'] > 100), 'age'] = np.nan
customers['age'] = customers['age'].fillna(customers.groupby('city')['age'].transform('median'))
# d. Convert signup_date
customers['signup_date'] = pd.to_datetime(customers['signup_date'])

# --- 3. CLEAN ORDERS ---
# a. Convert timestamp
orders['order_ts'] = pd.to_datetime(orders['order_ts'])
# b. Remove exact duplicates
orders = orders.drop_duplicates()
# c. Keep latest status per order_id
orders = orders.sort_values('order_ts').drop_duplicates(subset=['order_id'], keep='last')
# d. Clean order_amount
orders['order_amount'] = pd.to_numeric(orders['order_amount'], errors='coerce')
# Merge with products to calculate missing amounts
orders = orders.merge(products[['product_id', 'price_inr']], on='product_id', how='left')
calculated_amount = round(orders['price_inr'] * orders['quantity'] * (1 - orders['discount_pct'] / 100), 2)
orders['order_amount'] = orders['order_amount'].fillna(calculated_amount)
orders = orders.drop(columns=['price_inr'])
# e. Clean delivery_minutes
orders.loc[orders['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan
# Merge to get product category for median imputation
orders = orders.merge(products[['product_id', 'category']], on='product_id', how='left')
completed_mask = orders['status'] == 'completed'
median_delivery = orders[completed_mask].groupby('category')['delivery_minutes'].transform('median')
orders.loc[completed_mask & orders['delivery_minutes'].isna(), 'delivery_minutes'] = median_delivery[completed_mask & orders['delivery_minutes'].isna()]
orders = orders.drop(columns=['category'])

# --- 4. STORE TO CSV AND SQLITE ---
customers.to_csv(f"{CLEAN_DIR}/customers.csv", index=False)
products.to_csv(f"{CLEAN_DIR}/products.csv", index=False)
orders.to_csv(f"{CLEAN_DIR}/orders.csv", index=False)

conn = sqlite3.connect(DB_PATH)
# Store dates as text format 'YYYY-MM-DD HH:MM:SS'
orders_sql = orders.copy()
orders_sql['order_ts'] = orders_sql['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
customers_sql = customers.copy()
customers_sql['signup_date'] = customers_sql['signup_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

customers_sql.to_sql('customers', conn, if_exists='replace', index=False)
products.to_sql('products', conn, if_exists='replace', index=False)
orders_sql.to_sql('orders', conn, if_exists='replace', index=False)
conn.close()

print("Data Cleaning and Storage Complete.")