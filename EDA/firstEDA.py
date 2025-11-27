import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('../data/train.csv')

print("="*50)
print("       INTENSIVE DATA HEALTH CHECK       ")
print("="*50)

# 1. Basic Info
print("\n--- 1. SHAPE & TYPES ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.info())

# 2. Missing Values Deep Dive
print("\n--- 2. MISSING VALUES BREAKDOWN ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Percent %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percent %', ascending=False))

# 3. Duplicates
print("\n--- 3. DUPLICATE RECORDS ---")
print(f"Duplicate Rows: {df.duplicated().sum()}")

# 4. Cardinality (Unique Values)
print("\n--- 4. UNIQUE VALUES (CARDINALITY) ---")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# 5. Target Balance
print("\n--- 5. TARGET VARIABLE DISTRIBUTION ---")
print(df['retention_status'].value_counts(normalize=True) * 100)

# 6. Numerical Stats (Skewness check)
print("\n--- 6. NUMERICAL STATISTICS ---")
print(df.describe().T)

print("\n--- 7. CATEGORICAL BREAKDOWNS (Top 5 per col) ---")
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col != 'retention_status':
        print(f"\n[{col} Top 5 Categories]")
        print(df[col].value_counts().head(5))

print("\n" + "="*50)
print("       END OF STATISTICAL REPORT       ")
print("="*50)