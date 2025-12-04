import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# 1. Load Data
gj_df = pd.read_csv('raw_data/gj_final_matched.csv')
hs_df = pd.read_csv('raw_data/hs_final_matched.csv')
gj_df['Region'] = 'Gimje'
hs_df['Region'] = 'Hwaseong'
total_df = pd.concat([gj_df, hs_df], ignore_index=True)
clean_df = total_df[total_df['yield_weight'] >= 0.1].copy()

# Set style correctly
sns.set_style("whitegrid")


# 2. Nitrogen Requirement Calculation (RDA Formula)
def calculate_n_req(row):
    om = row['soil_OM']
    si = min(row['soil_AVSi'], 180)  # Cap Si at 180
    n_req = 9.14 - (0.109 * om) + (0.020 * si)
    return max(n_req, 0)


clean_df['Total_N_Req'] = clean_df.apply(calculate_n_req, axis=1)

# 3. Threshold Analysis & Visualization

fig = plt.figure(figsize=(20, 18))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# --- A. Protein Control ---

# A-1. Soil OM Threshold (Decision Tree)
tree_om = DecisionTreeRegressor(max_depth=1, random_state=42)
tree_om.fit(clean_df[['soil_OM']], clean_df['yield_protein'])
om_threshold = tree_om.tree_.threshold[0]

ax1 = fig.add_subplot(3, 2, 1)
sns.scatterplot(data=clean_df, x='soil_OM', y='yield_protein', hue='Region', s=80, ax=ax1, palette='viridis')
ax1.axhline(6.0, color='red', linestyle='-', linewidth=2, label='Protein Limit (6.0%)')
ax1.axvline(om_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold ({om_threshold:.1f})')
ax1.set_title(f'[Protein] Soil OM Threshold: {om_threshold:.1f} g/kg', fontsize=14, fontweight='bold')
ax1.set_xlabel('Soil Organic Matter (g/kg)')
ax1.set_ylabel('Protein (%)')
ax1.legend()

# A-2. Drone 02_GNDVI Threshold (Gimje)
gj_data = clean_df[clean_df['Region'] == 'Gimje'].copy()
tree_gndvi_gj = DecisionTreeRegressor(max_depth=1, random_state=42)
tree_gndvi_gj.fit(gj_data[['02_GNDVI']], gj_data['yield_protein'])
gj_gndvi_threshold = tree_gndvi_gj.tree_.threshold[0]

ax2 = fig.add_subplot(3, 2, 2)
sns.scatterplot(data=gj_data, x='02_GNDVI', y='yield_protein', color='green', s=80, ax=ax2)
ax2.axhline(6.0, color='red', linestyle='-', linewidth=2, label='Limit (6.0%)')
ax2.axvline(gj_gndvi_threshold, color='blue', linestyle='--', linewidth=2,
            label=f'Threshold ({gj_gndvi_threshold:.3f})')
ax2.set_title(f'[Protein/Gimje] 02_GNDVI Threshold: {gj_gndvi_threshold:.3f}', fontsize=14, fontweight='bold')
ax2.set_xlabel('02_GNDVI (July)')
ax2.set_ylabel('Protein (%)')
ax2.legend()

# --- B. Yield Security ---

# B-1. Soil Si Threshold (Poly Fit)
ax3 = fig.add_subplot(3, 2, 3)
sns.scatterplot(data=clean_df, x='soil_AVSi', y='yield_weight', hue='Region', s=80, ax=ax3, palette='viridis')

regions = ['Gimje', 'Hwaseong']
colors = ['green', 'orange']
si_thresholds = {}

for region, color in zip(regions, colors):
    reg_data = clean_df[clean_df['Region'] == region]
    if len(reg_data) > 2:
        z = np.polyfit(reg_data['soil_AVSi'], reg_data['yield_weight'], 2)
        p = np.poly1d(z)
        peak_si = -z[1] / (2 * z[0])
        si_thresholds[region] = peak_si

        x_range = np.linspace(reg_data['soil_AVSi'].min(), reg_data['soil_AVSi'].max(), 100)
        ax3.plot(x_range, p(x_range), color=color, linewidth=2, linestyle='-')
        ax3.axvline(peak_si, color=color, linestyle='--', label=f'{region} Peak ({peak_si:.0f})')

ax3.set_title('[Yield] Soil Si Optimal Thresholds (Poly Fit)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Available Silicate (ppm)')
ax3.set_ylabel('Yield (kg)')
ax3.legend()

# B-2. Potassium (K) Saturation (Poly Fit)
z_k = np.polyfit(clean_df['soil_K'], clean_df['yield_weight'], 2)
p_k = np.poly1d(z_k)
k_peak = -z_k[1] / (2 * z_k[0])

ax4 = fig.add_subplot(3, 2, 4)
sns.scatterplot(data=clean_df, x='soil_K', y='yield_weight', hue='Region', s=80, ax=ax4, palette='viridis')
x_range_k = np.linspace(clean_df['soil_K'].min(), clean_df['soil_K'].max(), 100)
ax4.plot(x_range_k, p_k(x_range_k), color='black', linewidth=2, label='Global Trend')
ax4.axvline(k_peak, color='red', linestyle='--', label=f'Saturation ({k_peak:.2f})')
ax4.set_title(f'[Yield] Potassium(K) Saturation Point: {k_peak:.2f}', fontsize=14, fontweight='bold')
ax4.set_xlabel('Soil K (cmol/kg)')
ax4.set_ylabel('Yield (kg)')
ax4.legend()

# --- C. Hwaseong Special ---

# C-1. Hwaseong 02_GNDVI Peak
hs_data = clean_df[clean_df['Region'] == 'Hwaseong']
z_hs_gndvi = np.polyfit(hs_data['02_GNDVI'], hs_data['yield_weight'], 2)
p_hs_gndvi = np.poly1d(z_hs_gndvi)
hs_gndvi_peak = -z_hs_gndvi[1] / (2 * z_hs_gndvi[0])

ax5 = fig.add_subplot(3, 2, 5)
sns.scatterplot(data=hs_data, x='02_GNDVI', y='yield_weight', color='orange', s=80, ax=ax5)
x_range_hs = np.linspace(hs_data['02_GNDVI'].min(), hs_data['02_GNDVI'].max(), 100)
ax5.plot(x_range_hs, p_hs_gndvi(x_range_hs), color='blue', linewidth=2, label='Trend')
ax5.axvline(hs_gndvi_peak, color='red', linestyle='--', label=f'Peak ({hs_gndvi_peak:.3f})')
ax5.set_title(f'[Yield/Hwaseong] 02_GNDVI Peak: {hs_gndvi_peak:.3f}', fontsize=14, fontweight='bold')
ax5.set_xlabel('02_GNDVI (July)')
ax5.set_ylabel('Yield (kg)')
ax5.legend()

# C-2. Text Summary
ax6 = fig.add_subplot(3, 2, 6)
ax6.axis('off')
text_str = f"""
[Summary of Logic & Formulas]

1. Nitrogen Requirement (RDA Standard)
   N = 9.14 - 0.109 * OM + 0.020 * Si
   (Constraint: Si max 180 ppm)

2. Threshold Logic
   A. Protein Control (Decision Tree)
      - Find split point maximizing variance reduction.
      - OM Threshold: {om_threshold:.1f} g/kg
      - Gimje GNDVI: {gj_gndvi_threshold:.3f}

   B. Yield Security (Polynomial Regression)
      - Find vertex (x = -b/2a) of quadratic curve.
      - Gimje Si Peak: {si_thresholds.get('Gimje', 0):.0f} mg/kg
      - Hwaseong Si Peak: {si_thresholds.get('Hwaseong', 0):.0f} mg/kg
      - Hwaseong GNDVI Peak: {hs_gndvi_peak:.3f}

3. Application
   - If Value > Threshold: Reduce/Stop Fertilizer
   - If Value < Threshold: Standard/Increase Fertilizer
"""
ax6.text(0.1, 0.5, text_str, fontsize=12, family='monospace', va='center')

plt.suptitle("Step 4. Threshold Logic Visualization & Formulas", fontsize=20, y=0.95)
plt.savefig('output/new_step/step4_threshold_logic_visualization_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

print("=== Threshold Calculation Results ===")
print(f"OM Threshold: {om_threshold:.4f}")
print(f"Gimje GNDVI Threshold: {gj_gndvi_threshold:.4f}")
print(f"Hwaseong GNDVI Peak: {hs_gndvi_peak:.4f}")
print(f"Si Peaks: {si_thresholds}")
print(f"K Peak: {k_peak:.4f}")