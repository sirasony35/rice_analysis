import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. 데이터 로드 및 전처리
gj_df = pd.read_csv('raw_data/gj_final_matched.csv')
hs_df = pd.read_csv('raw_data/hs_final_matched.csv')

gj_df['Region'] = 'Gimje'
hs_df['Region'] = 'Hwaseong'

total_df = pd.concat([gj_df, hs_df], ignore_index=True)
clean_df = total_df[total_df['yield_weight'] >= 0.1].copy()

# 2. 상관관계 분석 (전체 및 지역별)
def get_correlations(df, region_name="Global"):
    corr_k = df['soil_K'].corr(df['yield_weight'])
    corr_si = df['soil_AVSi'].corr(df['yield_weight'])
    return {'Region': region_name, 'Corr_K_Yield': corr_k, 'Corr_Si_Yield': corr_si}

global_corr = get_correlations(clean_df, "All")
gj_corr = get_correlations(clean_df[clean_df['Region'] == 'Gimje'], "Gimje")
hs_corr = get_correlations(clean_df[clean_df['Region'] == 'Hwaseong'], "Hwaseong")

corr_summary = pd.DataFrame([global_corr, gj_corr, hs_corr])

# 3. 시각화 (Scatter Plot with Regression Line)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Potassium (K) vs Yield
sns.scatterplot(data=clean_df, x='soil_K', y='yield_weight', hue='Region', style='Region', s=100, ax=axes[0], palette='Set2')
sns.regplot(data=clean_df, x='soil_K', y='yield_weight', scatter=False, color='gray', ax=axes[0]) # 전체 경향선
axes[0].set_title(f"Soil K vs Yield (Corr: {global_corr['Corr_K_Yield']:.2f})")
axes[0].set_xlabel("Soil Potassium (K) [cmol/kg]")
axes[0].set_ylabel("Yield Weight [kg]")

# Plot 2: Available Silicate (Si) vs Yield
sns.scatterplot(data=clean_df, x='soil_AVSi', y='yield_weight', hue='Region', style='Region', s=100, ax=axes[1], palette='Set2')
sns.regplot(data=clean_df, x='soil_AVSi', y='yield_weight', scatter=False, color='gray', ax=axes[1]) # 전체 경향선
axes[1].set_title(f"Soil Si vs Yield (Corr: {global_corr['Corr_Si_Yield']:.2f})")
axes[1].set_xlabel("Available Silicate (Si) [ppm]")
axes[1].set_ylabel("Yield Weight [kg]")

# 지역별 경향선 추가 (점선)
# 김제 Si 경향
sns.regplot(data=clean_df[clean_df['Region']=='Gimje'], x='soil_AVSi', y='yield_weight', scatter=False, ax=axes[1], color='green', line_kws={'linestyle':'--'}, ci=None)
# 화성 Si 경향
sns.regplot(data=clean_df[clean_df['Region']=='Hwaseong'], x='soil_AVSi', y='yield_weight', scatter=False, ax=axes[1], color='orange', line_kws={'linestyle':'--'}, ci=None)

plt.suptitle("Step 2-1: Impact of K & Si on Yield", fontsize=16)
plt.savefig('output/new_step/step2_1_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("=== 상관계수 분석 결과 (Pearson Correlation) ===")
print(corr_summary)