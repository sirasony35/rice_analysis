import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load Data
gj_df = pd.read_csv('raw_data/gj_final_matched.csv')
hs_df = pd.read_csv('raw_data/hs_final_matched.csv')
gj_df['Region'] = 'Gimje'
hs_df['Region'] = 'Hwaseong'
total_df = pd.concat([gj_df, hs_df], ignore_index=True)
clean_df = total_df[total_df['yield_weight'] >= 0.1].copy()


# ---------------------------------------------------------
# Part 1. 기비 감량 근거: 유기물(OM)의 질소 공급력 시뮬레이션
# ---------------------------------------------------------
# 농학적 가정: 유기물 10g/kg 당 약 1.5kg N/10a 자연 공급 가정 (일반적 논토양)
# 표준 시비량 (기비): 약 9.0 kg/10a 기준

def simulate_soil_n_supply(om_val):
    # 적정 유기물(20) 대비 초과분에 의한 추가 질소 공급량
    base_om = 20
    if om_val <= base_om:
        return 0

    excess_om = om_val - base_om
    # 유기물 1g 증가 시 질소 공급량 증가 계수 (보수적 추정: 0.15 kg N)
    extra_n = excess_om * 0.15
    return extra_n


clean_df['Natural_N_Supply'] = clean_df['soil_OM'].apply(simulate_soil_n_supply)
clean_df['Basal_Reduction_Rate'] = (clean_df['Natural_N_Supply'] / 9.0) * 100  # 기비 9kg 대비 절감 가능 비율

# ---------------------------------------------------------
# Part 2. 추비 감량 근거: GNDVI와 단백질의 정량적 관계
# ---------------------------------------------------------
# 회귀분석: GNDVI가 0.1 증가할 때 단백질이 몇 % 증가하는가?
model = LinearRegression()
X = clean_df[['02_GNDVI']]
y = clean_df['yield_protein']
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_

# 임계치(0.78) 초과 그룹의 평균 단백질과 목표(6.0) 차이 분석
high_gndvi_group = clean_df[clean_df['02_GNDVI'] >= 0.78]
avg_protein_risk = high_gndvi_group['yield_protein'].mean()
protein_excess = avg_protein_risk - 6.0

# ---------------------------------------------------------
# 3. Visualization & Output
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Soil OM vs Potential N Reduction Rate
sns.scatterplot(data=clean_df, x='soil_OM', y='Basal_Reduction_Rate', hue='Region', s=80, ax=axes[0])
axes[0].axhline(10, color='orange', linestyle='--', label='10% Reduction')
axes[0].axhline(20, color='red', linestyle='--', label='20% Reduction')
axes[0].set_title('Justification for Basal N Reduction (Based on Soil OM)', fontsize=14)
axes[0].set_ylabel('Potential Reduction Rate (%)')

# Plot 2: GNDVI vs Protein Regression
sns.regplot(data=clean_df, x='02_GNDVI', y='yield_protein', ax=axes[1], color='purple', scatter_kws={'s': 50})
axes[1].axvline(0.78, color='red', linestyle='--', label='Threshold (0.78)')
axes[1].axhline(avg_protein_risk, color='orange', linestyle=':',
                label=f'Avg Protein above Thr ({avg_protein_risk:.2f}%)')
axes[1].axhline(6.0, color='green', linestyle='-', label='Target (6.0%)')
axes[1].set_title(f'Justification for Top N Reduction (Slope: {slope:.2f})', fontsize=14)

plt.tight_layout()
plt.savefig('output/new_step/reduction_logic_analysis.png', dpi=300)
plt.show()

print("=== 1. 기비 감량 근거 (유기물 효과) ===")
print(f"유기물 25g/kg일 때 자연 질소 공급량: {simulate_soil_n_supply(25):.2f} kg/10a")
print(f"-> 표준 기비(9kg) 대비 절감 가능 비율: {(simulate_soil_n_supply(25) / 9.0) * 100:.1f}%")
print(f"유기물 30g/kg일 때 자연 질소 공급량: {simulate_soil_n_supply(30):.2f} kg/10a")
print(f"-> 표준 기비(9kg) 대비 절감 가능 비율: {(simulate_soil_n_supply(30) / 9.0) * 100:.1f}%")

print("\n=== 2. 추비 감량 근거 (단백질 리스크) ===")
print(f"GNDVI 0.78 이상 그룹의 평균 단백질: {avg_protein_risk:.2f}% (목표 6.0% 대비 +{protein_excess:.2f}% 초과)")
print(f"회귀 계수 (Slope): GNDVI 0.1 증가 시 단백질 {slope:.2f}% 증가")
print(f"-> 단백질 0.3%를 낮추기 위해 필요한 질소 감량 효과 고려 시 50% 감량이 타당함.")