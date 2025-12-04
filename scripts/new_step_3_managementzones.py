import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 데이터 로드 및 전처리
gj_df = pd.read_csv('raw_data/gj_final_matched.csv')
hs_df = pd.read_csv('raw_data/hs_final_matched.csv')

gj_df['Region'] = 'Gimje'
hs_df['Region'] = 'Hwaseong'

total_df = pd.concat([gj_df, hs_df], ignore_index=True)
clean_df = total_df[total_df['yield_weight'] >= 0.1].copy()

# 2. 군집화 변수 선정
# - 토양(원인): K(수확량), Si(과다체크), OM(기초체력), Mg(화성특성)
# - 드론(상태): 02_GNDVI (7월 시비시점), 04_GNDVI (9월 단백질 위험확인)
# - 결과(지표): 수확량, 단백질
features = ['soil_K', 'soil_AVSi', 'soil_OM', 'soil_Mg', '02_GNDVI', '04_GNDVI', 'yield_weight', 'yield_protein']
X = clean_df[features].copy()
X = X.fillna(X.mean())

# 3. 스케일링 및 K-Means 군집화 (K=4)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clean_df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. 군집별 특성 요약
cluster_summary = clean_df.groupby('Cluster')[features].mean().T
cluster_counts = clean_df['Cluster'].value_counts().sort_index()

# 5. 시각화 (수확량 vs 단백질)
plt.figure(figsize=(12, 8))
sns.scatterplot(data=clean_df, x='yield_weight', y='yield_protein', hue='Cluster', style='Region',
                palette='viridis', s=120, edgecolor='k')

# 군집 중심 특성 표시 (평균 위치에 텍스트)
summary_T = cluster_summary.T
for i in range(4):
    c_yield = summary_T.loc[i, 'yield_weight']
    c_prot = summary_T.loc[i, 'yield_protein']
    plt.text(c_yield, c_prot+0.1, f'Zone {i}',
             fontsize=12, fontweight='bold', ha='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round'))

# 기준선
plt.axhline(6.0, color='r', linestyle='--', label='Protein Limit (6.0%)')
plt.axvline(clean_df['yield_weight'].mean(), color='b', linestyle='--', label='Avg Yield')

plt.title('Step 3. 2026 Management Zones (Based on Soil + Drone + Yield)', fontsize=16)
plt.xlabel('Yield Weight (kg)')
plt.ylabel('Grain Protein (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.savefig('output/new_step/step3_re_clustering.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 결과 출력
print("=== Cluster Summary (Mean) ===")
print(cluster_summary)
print("\n=== Cluster Counts by Region ===")
print(pd.crosstab(clean_df['Region'], clean_df['Cluster']))