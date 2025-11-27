import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime


OUTPUT_DIR = "../EDA_Output/SecondEDAOutput"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving all plots to:", OUTPUT_DIR)

log_file = f"{OUTPUT_DIR}/EDA_console_output.txt"

class Tee:
    """Redirect stdout to both console and file."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, open(log_file, "w"))

print("Console logging started...")
print("Log file:", log_file)
print("-" * 60)


df = pd.read_csv('../data/train.csv')
print("Dataset Loaded. Shape:", df.shape)

sns.set(style="whitegrid")

# Filter out ID column
plot_df = df.drop(columns=['founder_id'], errors='ignore')

print("Preparing analysis...")


num_cols = plot_df.select_dtypes(include=['float64', 'int64']).columns
print("Numerical Columns:", list(num_cols))


# 1. Correlation Matrix
print("\nGenerating Correlation Heatmap...")
plt.figure(figsize=(12, 10))
sns.heatmap(plot_df[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Master Correlation Heatmap')
plt.savefig(f"{OUTPUT_DIR}/heatmap_correlation.png", dpi=300, bbox_inches='tight')
plt.show()


# 2. Distribution & Boxplots Loop
print("\nGenerating Distribution and Boxplots...")
for col in num_cols:
    print(f"Processing column: {col} ...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    sns.histplot(data=plot_df, x=col, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f"Distribution of {col}")
    fig.savefig(f"{OUTPUT_DIR}/dist_{col}.png", dpi=300, bbox_inches='tight')
    
    # Boxplot vs Target
    sns.boxplot(data=plot_df, x='retention_status', y=col, hue='retention_status',
                legend=False, palette='coolwarm', ax=axes[1])
    axes[1].set_title(f"{col} vs Retention Status")
    fig.savefig(f"{OUTPUT_DIR}/box_{col}.png", dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


cat_cols = plot_df.select_dtypes(include=['object']).columns
cat_cols = [c for c in cat_cols if c != 'retention_status']

print("\nCategorical Columns:", list(cat_cols))

for col in cat_cols:
    print(f"Processing categorical column: {col} ...")
    
    plt.figure(figsize=(10, 5))
    
    if plot_df[col].nunique() > 10:
        plt.xticks(rotation=45, ha='right')
        
    sns.countplot(data=plot_df, x=col, hue='retention_status', palette='viridis')
    plt.title(f"{col} Distribution by Retention Status")
    plt.legend(title='Retention', loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/countplot_{col}.png", dpi=300, bbox_inches='tight')
    plt.show()


print("\nAll EDA plots saved successfully!")
print("Console output saved to:", log_file)


sys.stdout = original_stdout
print("Logging completed.")