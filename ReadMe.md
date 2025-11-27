# Start-up Founder Retention Prediction

## Team Information

**Team Name:** Predictify  

**Team Members:**  
1. Kartikeya Dimri – IMT2023126  
2. Ayush Mishra – IMT2023129  
3. Harsh Sinha – IMT2023571  

---

## Overview

This project aims to predict whether a startup founder will **stay with** their venture or **exit**, based on demographic, behavioural, operational, and startup-level indicators.  
The task is framed as a **binary classification** problem with expected class imbalance, making **Macro F1 Score** the primary evaluation metric.

The repository includes:

- Multi-stage EDA (4 separate in-depth analyses)  
- Three major preprocessing pipelines (feature engineering, cleaning, interaction terms)  
- A large suite of ML models (NNs, SVMs, boosting, ensembles, classical ML)  
- Cross-validated experiments and hyperparameter tuning  
- Model interpretation and comparison  

This project reflects the practical challenges of churn prediction and early-warning analytics in startup ecosystems.  
:contentReference[oaicite:1]{index=1}

---

## Directory Structure



## Directory Structure

```
Start-up-Founder-Retention-Prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│
├── preprocessing/
│   ├── P1_preprocess.py
│   ├── P2_preprocess.ipynb
│   ├── P3_preprocess.ipynb
│   └── P4_preprocess(adv).ipynb
│
├── EDA/
│   ├── KnowingData.ipynb
│   ├── EDA_File_1.ipynb
│   ├── EDA_File_2.ipynb
│   ├── EDA_File_3.ipynb
│   └── EDA_File_4.ipynb
│
├── models/
│   ├── MLP_P1_721.py
│   ├── Neural_Lgbm_P4.py
│   ├── Lgbm1_P1_751.py
│   ├── Lgbm2WithMoreHyperParam_P1_740.py
│   ├── Cat_P1_738.py
│   ├── Bayes_MLP_Log_Lgbm_Xgb_Cat_P4.py
│   ├── AllModels_P2.ipynb
│   ├── AllModels_P3.ipynb
│   ├── Stack_P4.ipynb
│   └── Xg_LgbM_Cat_P1_736.ipynb
│
├── output/
│   └── *.csv
```


## Dataset Summary

The dataset contains **24 columns**, combining founder demographics, operational roles, satisfaction metrics, startup performance indicators, team structure, funding activity, and several behavioural attributes.  
The target variable is **retention_status** (Retained vs Exited).  
:contentReference[oaicite:2]{index=2}

Key characteristics:

- Mixed numerical, categorical, and ordinal variables  
- Significant behavioural indicators (work–life balance, burnout factors)  
- Multi-scale numeric fields (revenue, distance, age, years)  
- Non-uniform missing values  
- Mild but important class imbalance  

---

## Exploratory Data Analysis (EDA)

We conducted **four detailed EDA scripts**, each targeting specific insights. All of them are linked in the GitHub repository.

### **1. EDA File 1 — Data Health & Structure**  
Focus on:  
- Missing values  
- Duplicates  
- Feature cardinality  
- Numerical stats  
- Target distribution  
:contentReference[oaicite:3]{index=3}

### **2. EDA File 2 — Visual Behaviour of Numerical & Categorical Variables**  
Includes:  
- Heatmaps  
- Histograms & KDE  
- Boxplots vs target  
- Countplots for categorical variables  
:contentReference[oaicite:4]{index=4}

### **3. EDA File 3 — Statistical Diagnostics & Multivariate Interactions**  
Adds:  
- Skewness & kurtosis  
- Q–Q normality checks  
- Cramer’s V for categorical association strength  
- Multivariate pairplots, violin plots, and interaction visuals  
:contentReference[oaicite:5]{index=5}

### **4. EDA File 4 — Target-Centric Visualizations**  
Introduces:  
- Normalized stacked bar charts  
- Target-aware categorical comparison  
- Consolidated boxplots for fast inspection  
:contentReference[oaicite:6]{index=6}

---

## Preprocessing Pipelines

We built **Major preprocessing strategies**, each with different assumptions and feature transformations.

### **1. P1_preprocess.py — Baseline + Essential Feature Engineering**
Adds engineered metrics:
- *age_at_founding*  
- *tenure_ratio*  
- *unhappy_overtime*  

Handles:
- Missing values  
- Ordinal rating encodings  
- One-Hot encoding for nominal categories  
- Standard scaling for numerical features  

Outputs fully aligned train & test matrices.  
:contentReference[oaicite:7]{index=7}

---

### **2. Unified Cleaning + Consistent Cross-Dataset Preparation**
Key operations:
- Train+Test concatenation for uniform preprocessing  
- Feature “kill list” to remove noisy columns (IDs, visible attributes)  
- Log-transformation for skewed revenue  
- Multiple imputations (median/mode/Unknown)  
- Reconstructed train & test sets with identical transforms  
:contentReference[oaicite:8]{index=8}

---

### **3. Advanced Interaction-Based Engineering**
Introduces deeper behavioural & efficiency metrics:
- Revenue Efficiency  
- Founder Experience Gap  
- Burnout Index (overtime × satisfaction)  
- Additional binary/ordinal mappings  
- Feature space reduction  
:contentReference[oaicite:9]{index=9}

This pipeline maximizes model signal by constructing higher-order interactions tied to founder behaviour and startup maturity.

---

## Models Implemented

The project includes a variety of machine-learning families to compare performance:

### **Primary Models**
- **Neural Networks (MLP)** – Non-linear interactions, tuned architectures  
- **SVM (RBF Kernel)** – Margin-based separation for mixed-type data  
- **Logistic Regression** – Baseline linear performance  
:contentReference[oaicite:10]{index=10}

---

### **Secondary Models**
- **LightGBM (LGBM)** — Fast, powerful gradient boosting  
- **CatBoost** — Categorical-native boosting  
- **XGBoost** — Regularized boosting  
- **Random Forest**  
- **Gradient Boosting**  
- **Naive Bayes**  
- **Combined ensemble pipelines**  
:contentReference[oaicite:11]{index=11}

Several scripts and notebooks test tuned versions, voting/averaging ensembles, and mixed-model stacks.

---

## Results

LightGBM achieved the **highest Macro F1 = 0.751**, making it the best-performing model overall.

| Model Type                | Best File                     | Macro F1 |
|---------------------------|-------------------------------|----------|
| LightGBM                 | Lgbm1_P1_751.py               | **0.751** |
| CatBoost                 | Bayes_MLP_Log_Lgbm_Xgb_Cat    | 0.744    |
| Gradient Boosting        | AllModels_P2.ipynb            | 0.744    |
| Random Forest            | AllModels_P3.ipynb            | 0.735    |
| SVM (RBF)                | AllModels_P3.ipynb            | 0.733    |
| Neural Network (MLP)     | Neural_Lgbm_P4.py             | 0.732    |
| Naive Bayes              | Bayes_MLP_Log_Lgbm_Xgb_Cat    | 0.725    |
:contentReference[oaicite:12]{index=12}

---

## Key Observations

- **Boosting > NN > SVM > RF > Linear Models** on this dataset  
- Tree-based boosting handled mixed-type features and interactions best  
- Neural Networks competitive but sensitive to tuning  
- SVM strong but struggled with high-dimensional encoded categories  
- Outliers carried meaningful behavioural signal → removing them hurt performance  
- Macro F1 essential due to mild class imbalance  
:contentReference[oaicite:13]{index=13}

---

## Interpretation

**Why LightGBM won:**

- Works exceptionally well on structured tabular data  
- Handles heterogeneous, engineered feature space naturally  
- Boosting mechanism captures subtle behavioural and operational patterns  
- Less sensitive to scaling and distribution irregularities  
- Ensembles exploit weakly correlated features effectively  
:contentReference[oaicite:14]{index=14}

NNs and SVMs detected non-linear structure but could not outperform boosting due to dataset shape and categorical complexity.

---

