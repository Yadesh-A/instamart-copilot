import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# --- 1. Load Data ---
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

orders['order_ts'] = pd.to_datetime(orders['order_ts'])
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
orders = orders.merge(products[['product_id', 'category']], on='product_id', how='left')

# --- 2. TASK 4: Define the Lapse Problem ---
cutoff = pd.to_datetime('2025-09-01 00:00:00')

# Cohort: Customers with >=1 completed order BEFORE cutoff
before_cutoff = orders[orders['order_ts'] < cutoff].copy()
completed_before = before_cutoff[before_cutoff['status'] == 'completed']
cohort_cids = completed_before['customer_id'].unique()

# Label: Lapsed=1 if no completed order in Sep 2025
sep_orders = orders[(orders['order_ts'] >= cutoff) & (orders['order_ts'] < pd.to_datetime('2025-10-01 00:00:00'))]
sep_completed_cids = sep_orders[sep_orders['status'] == 'completed']['customer_id'].unique()

# Build Features (ONLY using data before cutoff)
features = []
for cid in cohort_cids:
    c_orders = before_cutoff[before_cutoff['customer_id'] == cid]
    c_completed = c_orders[c_orders['status'] == 'completed']
    c_cancelled = c_orders[c_orders['status'] == 'cancelled']
    
    recency = (cutoff - c_completed['order_ts'].max()).days
    frequency = c_completed['order_id'].nunique()
    monetary = c_completed['order_amount'].sum()
    avg_discount = c_completed['discount_pct'].mean()
    cancel_rate = len(c_cancelled) / (len(c_cancelled) + len(c_completed))
    avg_delivery = c_completed['delivery_minutes'].mean()
    fav_category = c_completed['category'].mode()[0] if not c_completed['category'].mode().empty else "Unknown"
    
    cust_info = customers[customers['customer_id'] == cid].iloc[0]
    tenure = (cutoff - cust_info['signup_date']).days
    
    lapsed = 0 if cid in sep_completed_cids else 1
    
    features.append({
        'customer_id': cid, 'recency': recency, 'frequency': frequency, 'monetary': monetary,
        'avg_discount': avg_discount, 'cancel_rate': cancel_rate, 'avg_delivery': avg_delivery,
        'fav_category': fav_category, 'city': cust_info['city'], 'age': cust_info['age'],
        'gender': cust_info['gender'], 'tenure': tenure, 'lapsed': lapsed
    })

df = pd.DataFrame(features)
df['avg_delivery'] = df['avg_delivery'].fillna(df['avg_delivery'].median())

print(f"Cohort Size: {len(df)}")
print(f"Lapsed Count: {df['lapsed'].sum()} ({(df['lapsed'].sum() / len(df)) * 100:.2f}%)")

X = df.drop(columns=['customer_id', 'lapsed'])
y = df['lapsed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- 3. TASK 5: Preprocess Without Leakage ---
num_cols = ['recency', 'frequency', 'monetary', 'avg_discount', 'cancel_rate', 'avg_delivery', 'age', 'tenure']
cat_cols = ['fav_category', 'city', 'gender']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

# Fit on train, transform test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Compare frequency mean
train_freq_mean = scaler_mean = preprocessor.named_transformers_['num'].mean_[1]
cohort_freq_mean = df['frequency'].mean()
print(f"\nFitted Scaler Frequency Mean: {train_freq_mean:.4f}")
print(f"Full Cohort Frequency Mean: {cohort_freq_mean:.4f}")

# --- 4. TASK 6: Train, Tune, Evaluate ---
print("\n=== Baseline ===")
y_pred_base = [0] * len(y_test)
print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred_base):.3f}")
print(f"Baseline F1 (Lapsed): {f1_score(y_test, y_pred_base, zero_division=0):.3f}")

# Logistic Regression Tuning
lr_grid = {'C': [0.01, 0.1, 1, 10, 100]}
lr_search = GridSearchCV(LogisticRegression(max_iter=1000), lr_grid, cv=5, scoring='f1')
lr_search.fit(X_train_processed, y_train)

# Random Forest Tuning
rf_grid = {'max_depth': [3, 5, 8], 'min_samples_leaf': [5, 10, 20]}
rf_search = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=42), rf_grid, cv=5, scoring='f1')
rf_search.fit(X_train_processed, y_train)

print(f"\nLR Best Params: {lr_search.best_params_}, CV F1: {lr_search.best_score_:.3f}")
print(f"RF Best Params: {rf_search.best_params_}, CV F1: {rf_search.best_score_:.3f}")

best_model = rf_search.best_estimator_
y_pred_rf = best_model.predict(X_test_processed)
print(f"\nTest Set Metrics (RF - Best Model):")
print(confusion_matrix(y_test, y_pred_rf))
print(f"Precision: {precision_score(y_test, y_pred_rf):.3f}, Recall: {recall_score(y_test, y_pred_rf):.3f}, F1: {f1_score(y_test, y_pred_rf):.3f}")

# --- 5. TASK 7: Class-Balance Experiment (Oversampling manually) ---
from sklearn.utils import resample
train_df = pd.DataFrame(X_train_processed)
train_df['lapsed'] = y_train.values

majority = train_df[train_df.lapsed == 0]
minority = train_df[train_df.lapsed == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
upsampled_train = pd.concat([majority, minority_upsampled])

X_train_bal = upsampled_train.drop('lapsed', axis=1)
y_train_bal = upsampled_train['lapsed']

lr_bal = LogisticRegression(C=lr_search.best_params_['C'], max_iter=1000)
lr_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = lr_bal.predict(X_test_processed)

print("\n=== Class Balance Experiment (Logistic Regression) ===")
print(f"Original Recall: {recall_score(y_test, lr_search.best_estimator_.predict(X_test_processed)):.3f}")
print(f"Balanced Recall: {recall_score(y_test, y_pred_bal):.3f}")
print(f"Original Precision: {precision_score(y_test, lr_search.best_estimator_.predict(X_test_processed)):.3f}")
print(f"Balanced Precision: {precision_score(y_test, y_pred_bal):.3f}")

# --- 6. TASK 8: Save and Output Retention List ---
full_pipeline = Pipeline([('preprocessor', preprocessor), ('model', best_model)])
full_pipeline.fit(X_train, y_train)

joblib.dump(full_pipeline, "outputs/lapse_model.joblib")
print("\nModel saved to outputs/lapse_model.joblib")

segments = pd.read_csv("outputs/customer_segments.csv")
premium_cids = segments[segments['is_premium'] == True]['customer_id'].tolist()

test_customers = df.loc[X_test.index, ['customer_id', 'lapsed']].copy()
test_customers['lapse_probability'] = full_pipeline.predict_proba(X_test)[:, 1]

retention_list = test_customers[test_customers['customer_id'].isin(premium_cids)].sort_values(by='lapse_probability', ascending=False)
retention_list = retention_list[['customer_id', 'lapse_probability', 'lapsed']].rename(columns={'lapsed': 'actually_lapsed'}).head(10)
retention_list.to_csv("outputs/retention_list.csv", index=False)
print("Retention list saved to outputs/retention_list.csv")
