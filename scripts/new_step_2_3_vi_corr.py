import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Data & Preprocessing
gj_df = pd.read_csv('raw_data/gj_final_matched.csv')
hs_df = pd.read_csv('raw_data/hs_final_matched.csv')

gj_df['Region'] = 'Gimje'
hs_df['Region'] = 'Hwaseong'

total_df = pd.concat([gj_df, hs_df], ignore_index=True)
clean_df = total_df[total_df['yield_weight'] >= 0.1].copy()

# ---------------------------------------------------------
# Part 1: Analyze 03_GNDVI vs Yield (Why Importance != Correlation?)
# ---------------------------------------------------------

# Calculate Correlation
corr_val = clean_df['03_GNDVI'].corr(clean_df['yield_weight'])

# Visualization: Scatter Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 1-1. Scatter Plot with linear regression
sns.regplot(data=clean_df, x='03_GNDVI', y='yield_weight', ax=axes[0], color='purple', scatter_kws={'s':50, 'alpha':0.6})
axes[0].set_title(f'Linear View: 03_GNDVI vs Yield (Corr: {corr_val:.2f})', fontsize=14)
axes[0].text(clean_df['03_GNDVI'].min(), clean_df['yield_weight'].max(),
             "Low Correlation implies weak LINEAR relationship,\nbut Random Forest captures NON-LINEAR patterns.",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# 1-2. LOWESS (Locally Weighted Scatterplot Smoothing) to see non-linear trend
sns.regplot(data=clean_df, x='03_GNDVI', y='yield_weight', lowess=True, ax=axes[1], color='darkblue', scatter_kws={'s':50, 'alpha':0.6})
axes[1].set_title('Non-Linear View (LOWESS Trend)', fontsize=14)

# ---------------------------------------------------------
# Part 2: Full Correlation Heatmap (All Sessions, All Indices)
# ---------------------------------------------------------

# Identify all drone index columns (01_~06_ and indices)
drone_indices = ['NDVI', 'GNDVI', 'NDRE', 'LCI', 'OSAVI']
drone_cols = [col for col in clean_df.columns if any(idx in col for idx in drone_indices) and col[0].isdigit()]
drone_cols.sort() # Sort by session (01, 02...)

# Calculate Correlation Matrix
target_vars = ['yield_weight', 'yield_protein']
corr_matrix = clean_df[drone_cols + target_vars].corr()

# Extract only correlations with targets
target_corr_matrix = corr_matrix.loc[drone_cols, target_vars]

# Plot Heatmap
plt.figure(figsize=(10, 15))
sns.heatmap(target_corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Heatmap: All Drone Indices vs Yield/Protein', fontsize=16)
plt.ylabel('Drone Indices (Session_Index)')
plt.tight_layout()
plt.savefig('output/new_step/step2_3_full_correlation_heatmap.png', dpi=300)

plt.show()

print(f"=== 03_GNDVI Correlation: {corr_val:.4f} ===")