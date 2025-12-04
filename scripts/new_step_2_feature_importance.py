import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# 1. Load Data
gj_df = pd.read_csv('raw_data/gj_final_matched.csv')
hs_df = pd.read_csv('raw_data/hs_final_matched.csv')

gj_df['Region'] = 'Gimje'
hs_df['Region'] = 'Hwaseong'

total_df = pd.concat([gj_df, hs_df], ignore_index=True)
clean_df = total_df[total_df['yield_weight'] >= 0.1].copy()

# 2. Define Variable Groups
# Determinants: Soil Factors
soil_vars = ['soil_pH', 'soil_EC', 'soil_OM', 'soil_AVP', 'soil_AVSi', 'soil_K', 'soil_Ca', 'soil_Mg']

# Predictors: Drone Indices
drone_indices = ['NDVI', 'GNDVI', 'NDRE', 'LCI', 'OSAVI']
drone_cols = [col for col in clean_df.columns if any(idx in col for idx in drone_indices) and col[0].isdigit()]

# Targets
target_yield = 'yield_weight'
target_protein = 'yield_protein'


# 3. Helper Function for Feature Importance
def get_feature_importance(df, features, target):
    X = df[features]
    y = df[target]

    # Handle missing values if any
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_imputed, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    feature_ranking = pd.DataFrame({
        'Feature': [features[i] for i in indices],
        'Importance': importances[indices]
    })
    return feature_ranking


# 4. Run Analysis
# 4-1. Yield Determinants (Soil)
fi_yield_soil = get_feature_importance(clean_df, soil_vars, target_yield)
# 4-2. Yield Predictors (Drone)
fi_yield_drone = get_feature_importance(clean_df, drone_cols, target_yield)

# 4-3. Protein Determinants (Soil)
fi_protein_soil = get_feature_importance(clean_df, soil_vars, target_protein)
# 4-4. Protein Predictors (Drone)
fi_protein_drone = get_feature_importance(clean_df, drone_cols, target_protein)

# 5. Visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)


# Function for bar plot
def plot_importance(ax, data, title, color_palette):
    sns.barplot(x='Importance', y='Feature', data=data.head(10), ax=ax, palette=color_palette)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Relative Importance')
    ax.set_ylabel('')


# Plotting
plot_importance(axes[0, 0], fi_yield_soil, 'Yield Determinants (Soil Factors)', 'Greens_r')
plot_importance(axes[0, 1], fi_yield_drone, 'Yield Predictors (Drone Indices)', 'Blues_r')
plot_importance(axes[1, 0], fi_protein_soil, 'Protein Determinants (Soil Factors)', 'Oranges_r')
plot_importance(axes[1, 1], fi_protein_drone, 'Protein Predictors (Drone Indices)', 'Purples_r')

plt.suptitle('Comprehensive Feature Importance Analysis: Determinants vs Predictors', fontsize=18)
plt.savefig('output/new_step/step2_refined_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Print Top Features
print("=== [Yield] Top Determinants (Soil) ===")
print(fi_yield_soil.head(5))
print("\n=== [Yield] Top Predictors (Drone) ===")
print(fi_yield_drone.head(5))
print("\n=== [Protein] Top Determinants (Soil) ===")
print(fi_protein_soil.head(5))
print("\n=== [Protein] Top Predictors (Drone) ===")
print(fi_protein_drone.head(5))