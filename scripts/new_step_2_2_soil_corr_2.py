import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
gj_df = pd.read_csv('raw_data/gj_final_matched.csv')
hs_df = pd.read_csv('raw_data/hs_final_matched.csv')

gj_df['Region'] = 'Gimje'
hs_df['Region'] = 'Hwaseong'

total_df = pd.concat([gj_df, hs_df], ignore_index=True)
clean_df = total_df[total_df['yield_weight'] >= 0.1].copy()


def find_optimal_threshold(df, region_name):
    region_data = df[df['Region'] == region_name].copy()

    # 2nd degree polynomial fit: y = ax^2 + bx + c
    z = np.polyfit(region_data['soil_AVSi'], region_data['yield_weight'], 2)
    p = np.poly1d(z)

    # Vertex (x = -b / 2a)
    vertex_x = -z[1] / (2 * z[0])

    # Check curvature
    concavity = "Concave Down (Max)" if z[0] < 0 else "Concave Up (Min)"

    # Determine Threshold
    # If concave down, the vertex is the peak yield (optimal Si).
    # If the calculated peak is within a reasonable range of the data, use it.
    data_min = region_data['soil_AVSi'].min()
    data_max = region_data['soil_AVSi'].max()

    if z[0] < 0 and data_min <= vertex_x <= data_max:
        threshold = vertex_x
    else:
        # Fallback: if curve is monotonic or peak is outside,
        # look for where the trend starts to flatten or decline significantly using sliding window or just use the vertex even if slightly outside to indicate trend.
        # For this specific request, let's strictly use the vertex if it implies a turning point.
        # If z[0] > 0 (convex), it means yield increases at extremes? Unlikely for fertilizer.
        # Let's trust the vertex but clamp it to data bounds if needed for visualization.
        threshold = vertex_x

    return region_data, threshold, p, z


# Analyze Gimje
gj_data, gj_threshold, gj_p, gj_z = find_optimal_threshold(clean_df, 'Gimje')

# Analyze Hwaseong
hs_data, hs_threshold, hs_p, hs_z = find_optimal_threshold(clean_df, 'Hwaseong')

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 6))


# Function to plot
def plot_analysis(ax, data, threshold, p, region_name, color_code):
    sns.scatterplot(data=data, x='soil_AVSi', y='yield_weight', color=color_code, s=100, ax=ax, label='Data Points')

    # Plot Trend Line
    x_range = np.linspace(data['soil_AVSi'].min(), data['soil_AVSi'].max(), 100)
    ax.plot(x_range, p(x_range), color='blue', linewidth=2, linestyle='-', label='Trend (Poly deg=2)')

    # Plot Threshold Line
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Peak Threshold ({threshold:.0f} ppm)')

    # Calculate Correlations
    under = data[data['soil_AVSi'] < threshold]
    over = data[data['soil_AVSi'] >= threshold]

    corr_under = under['soil_AVSi'].corr(under['yield_weight']) if len(under) > 2 else 0
    corr_over = over['soil_AVSi'].corr(over['yield_weight']) if len(over) > 2 else 0

    # Annotations
    y_pos = data['yield_weight'].max()
    ax.text(data['soil_AVSi'].min(), y_pos,
            f"Under {threshold:.0f} ppm:\nCorr = {corr_under:.2f}\nCount = {len(under)}",
            fontsize=12, color='blue', verticalalignment='top')

    ax.text(threshold + 10, y_pos,
            f"Over {threshold:.0f} ppm:\nCorr = {corr_over:.2f}\nCount = {len(over)}",
            fontsize=12, color='red', verticalalignment='top')

    ax.set_title(f'{region_name}: Optimal Si Threshold Analysis', fontsize=16)
    ax.set_xlabel('Available Silicate (Si) [ppm]')
    ax.set_ylabel('Yield Weight [kg]')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)


plot_analysis(axes[0], gj_data, gj_threshold, gj_p, 'Gimje', 'green')
plot_analysis(axes[1], hs_data, hs_threshold, hs_p, 'Hwaseong', 'orange')

plt.suptitle("Step 2-2: Data-Driven Optimal Silicate (Si) Thresholds", fontsize=20)
plt.tight_layout()
plt.savefig('output/new_step/step_2_2_optimal_thresholds.png', dpi=300)
plt.show()

print(f"=== Gimje Analysis ===")
print(f"Optimal Threshold: {gj_threshold:.2f} ppm")
print(f"Curve shape (a): {gj_z[0]:.6f} ({'Concave Down (Max)' if gj_z[0] < 0 else 'Concave Up (Min)'})")

print(f"\n=== Hwaseong Analysis ===")
print(f"Optimal Threshold: {hs_threshold:.2f} ppm")
print(f"Curve shape (a): {hs_z[0]:.6f} ({'Concave Down (Max)' if hs_z[0] < 0 else 'Concave Up (Min)'})")