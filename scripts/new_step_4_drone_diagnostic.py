import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Data
gj_df = pd.read_csv('raw_data/gj_final_matched.csv')
hs_df = pd.read_csv('raw_data/hs_final_matched.csv')
total_df = pd.concat([gj_df, hs_df], ignore_index=True)
clean_df = total_df[total_df['yield_weight'] >= 0.1].copy()

# Set style
sns.set_style("whitegrid")

# 2. Define Variables
determinants = ['leaf_N1', 'leaf_N2']
drone_indices = ['NDVI', 'GNDVI', 'NDRE', 'LCI', 'OSAVI']
drone_cols = [col for col in clean_df.columns if any(idx in col for idx in drone_indices) and col[0].isdigit()]
drone_cols.sort()
targets = ['yield_weight', 'yield_protein']

# --- Plot 1: Heatmap ---
plt.figure(figsize=(12, 10))
analysis_cols = drone_cols + targets + determinants
corr_matrix = clean_df[analysis_cols].corr()
heatmap_data = corr_matrix.loc[drone_cols, targets + determinants]
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0, fmt='.2f', cbar_kws={'label': 'Correlation'})
plt.title('1. Diagnostic Power of Drone Indices (Correlation Heatmap)', fontsize=16, fontweight='bold')
plt.ylabel('Drone Indices (Time)')
plt.tight_layout()
plt.savefig('output/new_step/step4_plot1_heatmap.png', dpi=300)
plt.close()

# --- Calculations for Plot 2 (Correlation Evolution) ---
sessions = sorted(list(set([c[:2] for c in drone_cols])))
corr_evolution = []
for session in sessions:
    cols = [c for c in drone_cols if c.startswith(session)]
    max_corr_n1 = clean_df[cols].apply(lambda x: x.corr(clean_df['leaf_N1'])).max()
    max_corr_n2 = clean_df[cols].apply(lambda x: x.corr(clean_df['leaf_N2'])).max()
    corr_evolution.append({'Session': session, 'Max_Corr': max_corr_n1, 'Target': 'Leaf_N1 (July)'})
    corr_evolution.append({'Session': session, 'Max_Corr': max_corr_n2, 'Target': 'Leaf_N2 (Aug)'})
corr_evo_df = pd.DataFrame(corr_evolution)

# --- Calculations for Plot 3 (Leaf N1) ---
r_early = clean_df['leaf_N1'].corr(clean_df['02_GNDVI'])
r_late = clean_df['leaf_N1'].corr(clean_df['06_GNDVI'])

# --- Calculations for Plot 4 (Leaf N2) ---
r_early_2 = clean_df['leaf_N2'].corr(clean_df['03_GNDVI'])
r_late_2 = clean_df['leaf_N2'].corr(clean_df['04_NDRE'])

# 3. Combined Visualization (Plots 2, 3, 4)
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
plt.subplots_adjust(wspace=0.3)

# Plot 2: Correlation Evolution
sns.lineplot(data=corr_evo_df, x='Session', y='Max_Corr', hue='Target', marker='o', linewidth=3, ax=axes[0], palette=['green', 'blue'])
axes[0].set_title('2. Lag Effect: When is Leaf N revealed?', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Max Correlation')
axes[0].grid(True, linestyle='--')

# Plot 3: Leaf N1 Scatter
sns.regplot(data=clean_df, x='leaf_N1', y='02_GNDVI', ax=axes[1], color='lightgreen', scatter_kws={'alpha':0.5}, label=f'02_GNDVI (July, r={r_early:.2f})')
sns.regplot(data=clean_df, x='leaf_N1', y='06_GNDVI', ax=axes[1], color='darkgreen', scatter_kws={'alpha':0.5}, label=f'06_GNDVI (Sep, r={r_late:.2f})')
axes[1].set_title('3. Leaf N1: Early Signal vs Late Result', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Drone Index')
axes[1].legend()

# Plot 4: Leaf N2 Scatter
sns.regplot(data=clean_df, x='leaf_N2', y='03_GNDVI', ax=axes[2], color='lightblue', scatter_kws={'alpha':0.5}, label=f'03_GNDVI (Aug, r={r_early_2:.2f})')
sns.regplot(data=clean_df, x='leaf_N2', y='04_NDRE', ax=axes[2], color='darkblue', scatter_kws={'alpha':0.5}, label=f'04_NDRE (Sep, r={r_late_2:.2f})')
axes[2].set_title('4. Leaf N2: Mid Signal vs Late Result', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Drone Index')
axes[2].legend()

plt.suptitle("Step 4 (Details). Lag Effect Analysis Combined (Plots 2-4)", fontsize=18, y=1.05)
plt.savefig('output/new_step/step4_plots_2_3_4_combined.png', dpi=300, bbox_inches='tight')
plt.show()