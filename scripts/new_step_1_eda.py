import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리 (Data Loading & Preprocessing)
# ---------------------------------------------------------
# CSV 파일 읽기
gj_df = pd.read_csv('raw_data/gj_final_matched.csv')
hs_df = pd.read_csv('raw_data/hs_final_matched.csv')

# 지역 구분 컬럼 추가
gj_df['Region'] = 'Gimje'
hs_df['Region'] = 'Hwaseong'

# 데이터 통합
total_df = pd.concat([gj_df, hs_df], ignore_index=True)

# 데이터 건전성 확보 (Cleaning)
# 수확량(yield_weight)이 0.1 미만인 비정상 데이터 제거
clean_df = total_df[total_df['yield_weight'] >= 0.1].copy()

print(f"Original Data Size: {len(total_df)}")
print(f"Cleaned Data Size: {len(clean_df)}")

# ---------------------------------------------------------
# 2. 분석 대상 변수 정의 (Variable Definition)
# ---------------------------------------------------------
# 전체 토양 분석 항목 (8종)
all_soil_vars = [
    'soil_pH',   # 산도 (적정: 6.0~6.5)
    'soil_EC',   # 전기전도도 (염류농도, 2.0 이하 권장)
    'soil_OM',   # 유기물 (지력, 20~30 권장)
    'soil_AVP',  # 유효인산 (비료 과다 지표)
    'soil_AVSi', # 유효규산 (벼 필수 양분, 200 이상 권장)
    'soil_K',    # 칼륨
    'soil_Ca',   # 칼슘 (세포벽 강화)
    'soil_Mg'    # 마그네슘 (광합성)
]

# ---------------------------------------------------------
# 3. 시각화: 지역별 토양 환경 비교 (Boxplot Visualization)
# ---------------------------------------------------------
# 2행 4열(총 8개) 서브플롯 생성
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten() # 반복문 처리를 위해 1차원으로 변환

# 반복문으로 모든 토양 변수 그리기
for i, col in enumerate(all_soil_vars):
    sns.boxplot(data=clean_df, x='Region', y=col, ax=axes[i], palette='Set2')
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

    # [참고] 농학적 주요 기준선 표시 (Reference Lines)
    if col == 'soil_pH':
        axes[i].axhline(6.5, color='red', linestyle='--', alpha=0.5, label='Max pH')
    elif col == 'soil_EC':
        axes[i].axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Salt Limit')
    elif col == 'soil_AVSi':
        axes[i].axhline(200, color='blue', linestyle='--', alpha=0.5, label='Min Si')

# 레이아웃 조정 및 제목 설정
plt.tight_layout()
plt.suptitle('Comparison of All Soil Properties (Gimje vs Hwaseong)', y=1.02, fontsize=16)

# [추가됨] 그래프를 이미지 파일로 저장
plt.savefig('output/new_step/step1_soil_comparison.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'step1_soil_comparison.png'")

# 화면에 출력
plt.show()

# ---------------------------------------------------------
# 4. 통계 요약 보고서 (Statistical Summary)
# ---------------------------------------------------------
# 지역별 평균(Mean)과 표준편차(Std) 계산
summary_stats = clean_df.groupby('Region')[all_soil_vars].agg(['mean', 'std']).round(2).T

print("\n=== [Result] 지역별 토양 성분 정밀 비교 ===")
print(summary_stats)