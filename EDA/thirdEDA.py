import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import os
import sys

# -----------------------------------------------------
# CREATE FOLDER FOR OUTPUTS
# -----------------------------------------------------
OUTPUT_DIR = "../EDA_Output/ThirdEDAOutput"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving all plots to:", OUTPUT_DIR)

# -----------------------------------------------------
# SETUP CONSOLE LOGGING
# -----------------------------------------------------
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

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
sns.set(style="whitegrid")
DATA_PATH = '../data/train.csv'

try:
    print("Loading Dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data Loaded Successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}. Please check.")
    sys.stdout = original_stdout
    raise

# ==============================================================================
# PART 1: NUMERICAL DISTRIBUTION & SKEW (The "MRI")
# ==============================================================================
print("\n" + "="*60)
print("      PART 1: NUMERICAL DISTRIBUTION & SKEW ANALYSIS")
print("="*60)

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
num_cols = [c for c in num_cols if c != 'founder_id']

dist_summary = pd.DataFrame(index=num_cols)
dist_summary['Skewness'] = df[num_cols].skew()
dist_summary['Kurtosis'] = df[num_cols].kurt()
dist_summary['Null_Count'] = df[num_cols].isnull().sum()

def recommend_transform(row):
    if abs(row['Skewness']) > 1:
        return "Log/Power Transform Needed (High Skew)"
    elif abs(row['Skewness']) > 0.5:
        return "Moderate Skew (Consider Scaling)"
    else:
        return "Normal-ish (StandardScaler ok)"

dist_summary['Recommendation'] = dist_summary.apply(recommend_transform, axis=1)

print("\nDISTRIBUTION HEALTH CHECK:")
print(dist_summary.sort_values(by='Skewness', key=abs, ascending=False))

# --- VISUALIZATION OF SKEW ---
skewed_cols = dist_summary[dist_summary['Recommendation'].str.contains("Transform")].index.tolist()

if len(skewed_cols) > 0:
    print(f"\n[Visualizing Top Skewed Features]: {skewed_cols}")
    
    rows = len(skewed_cols)
    plt.figure(figsize=(14, 5 * rows))

    for i, col in enumerate(skewed_cols):
        # Histogram
        ax1 = plt.subplot(rows, 2, i*2 + 1)
        sns.histplot(df[col].dropna(), kde=True, color='purple', ax=ax1)
        ax1.set_title(f'{col} Distribution (Skew: {df[col].skew():.2f})')

        # QQ plot
        ax2 = plt.subplot(rows, 2, i*2 + 2)
        stats.probplot(df[col].dropna(), dist="norm", plot=ax2)
        ax2.set_title(f'{col} Q-Q Plot')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/skew_plots.png", dpi=300, bbox_inches='tight')
    plt.show()

else:
    print("\nNo heavily skewed columns found (Skewness < 1.0).")


# ==============================================================================
# PART 2: CATEGORICAL PREDICTIVE POWER (Cramer's V)
# ==============================================================================
print("\n" + "="*60)
print("      PART 2: CATEGORICAL FEATURE STRENGTH (CRAMER'S V)")
print("="*60)

def cramers_v(x, y):
    cm = pd.crosstab(x, y)
    chi2 = chi2_contingency(cm)[0]
    n = cm.sum().sum()
    phi2 = chi2 / n
    r, k = cm.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0.0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
correlations = {}

print("Calculating correlations...")
for col in cat_cols:
    if col != 'retention_status':
        clean_data = df[[col, 'retention_status']].dropna()
        if not clean_data.empty:
            correlations[col] = cramers_v(clean_data[col], clean_data['retention_status'])

cat_corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Cramers_V'])
cat_corr_df = cat_corr_df.sort_values(by='Cramers_V', ascending=False)

print("\nSTRENGTH OF ASSOCIATION:")
print(cat_corr_df)

# Barplot
plt.figure(figsize=(10, 8))
sns.barplot(
    x=cat_corr_df.Cramers_V,
    y=cat_corr_df.index,
    hue=cat_corr_df.index,
    legend=False,
    palette='magma'
)
plt.axvline(x=0.05, color='red', linestyle='--')
plt.title("Categorical Predictive Strength (Cramer's V)")
plt.xlabel("Cramer's V")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cramers_v_barplot.png", dpi=300, bbox_inches='tight')
plt.show()


# ==============================================================================
# PART 3: MULTIVARIATE INTERACTIONS
# ==============================================================================
print("\n" + "="*60)
print("      PART 3: MULTIVARIATE INTERACTIONS & PATTERNS")
print("="*60)

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='founder_age',
    y='monthly_revenue_generated',
    hue='retention_status',
    alpha=0.6,
    palette='coolwarm'
)
plt.title('Revenue vs Age by Retention')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/scatter_age_revenue.png", dpi=300, bbox_inches='tight')
plt.show()

# Violin plot
if 'venture_satisfaction' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=df,
        x='venture_satisfaction',
        y='monthly_revenue_generated',
        hue='retention_status',
        split=True,
        order=['Low', 'Medium', 'High', 'Very High'],
        palette='muted'
    )
    plt.title('Revenue by Venture Satisfaction & Retention')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/violin_satisfaction_revenue.png", dpi=300, bbox_inches='tight')
    plt.show()

# Pairplot
plot_cols = ['founder_age', 'years_with_startup', 'monthly_revenue_generated', 'retention_status']
plot_cols = [c for c in plot_cols if c in df.columns]

if len(plot_cols) > 1:
    print("Generating Pairplot… (may take time)")
    sns.pairplot(df[plot_cols], hue='retention_status', palette='husl', corner=True)
    plt.suptitle("Pairwise Key Numerical Relationships", y=1.02)
    plt.savefig(f"{OUTPUT_DIR}/pairplot_key_features.png", dpi=300, bbox_inches='tight')
    plt.show()

print("\n" + "="*60)
print("      EXTENSIVE EDA COMPLETE")
print("="*60)

# -----------------------------------------------------
# RESTORE STDOUT
# -----------------------------------------------------
sys.stdout = original_stdout
print("Logging complete. Console output saved to:", log_file)