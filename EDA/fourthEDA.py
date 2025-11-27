import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys



OUTPUT_DIR = "../EDA_Output/FourthEDAOutput"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving all plots to:", OUTPUT_DIR)


log_file = f"{OUTPUT_DIR}/EDA_console_output.txt"

class Tee:
    """Redirect stdout to both notebook and a text file."""
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


sns.set(style="whitegrid")
plt.rcParams['figure.max_open_warning'] = 50

possible_paths = ['../data/train.csv', 'train.csv']
DATA_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        DATA_PATH = path
        break

if DATA_PATH is None:
    print("❌ Error: 'train.csv' not found.")
else:
    print(f"✅ Data Loaded from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

   
   
    print("\n" + "="*60)
    print("      1. MASTER CORRELATION HEATMAP")
    print("="*60)

    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if 'founder_id' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['founder_id'])

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))

    sns.heatmap(
        numeric_df.corr(),
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        linewidths=0.5,
        vmin=-1, vmax=1
    )
    plt.title('Correlation Matrix of All Numerical Features')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_correlation.png", dpi=300, bbox_inches='tight')
    plt.show()

  
    print("\n" + "="*60)
    print("      2. NUMERIC FEATURES vs TARGET")
    print("="*60)

    num_cols = numeric_df.columns.tolist()

    n_cols = 3
    n_rows = (len(num_cols) - 1) // n_cols + 1
    plt.figure(figsize=(15, 4 * n_rows))

    for i, col in enumerate(num_cols):
        ax = plt.subplot(n_rows, n_cols, i+1)

        sns.boxplot(
            data=df,
            x='retention_status',
            y=col,
            hue='retention_status',
            palette='Set2',
            legend=False,
            ax=ax
        )
        ax.set_title(f'{col} vs Retention')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/boxplots_numeric_vs_target.png", dpi=300, bbox_inches='tight')
    plt.show()

  
    print("\n" + "="*60)
    print("      3. CATEGORICAL FEATURES vs TARGET")
    print("="*60)

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'retention_status']
    cat_cols = [c for c in cat_cols if df[c].nunique() < 20]

    for col in cat_cols:
        print(f"Processing {col}...")

        cross_tab = pd.crosstab(df[col], df['retention_status'])
        cross_tab_prop = cross_tab.div(cross_tab.sum(1), axis=0)

        ax = cross_tab_prop.plot(
            kind='bar',
            stacked=True,
            figsize=(10, 4),
            colormap='viridis',
            edgecolor='black'
        )

        plt.title(f'Retention Rates by {col} (Normalized)')
        plt.xlabel(col)
        plt.ylabel('Proportion')
        plt.legend(title='Retention Status', bbox_to_anchor=(1.05, 1), loc='upper left')

        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            if height > 0.05:
                ax.text(x + width/2, y + height/2, f"{height*100:.0f}%",
                        ha='center', va='center', color='white', weight='bold')

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/stackedbar_{col}.png", dpi=300, bbox_inches='tight')
        plt.show()

print("\n" + "="*60)
print("      FOURTH EDA COMPLETE")
print("="*60)


sys.stdout = original_stdout
print("Logging finished. Console output saved to:", log_file)