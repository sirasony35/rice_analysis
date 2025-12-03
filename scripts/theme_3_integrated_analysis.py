import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# [설정] 파일 경로
# ==========================================
FILE_GJ = 'output/gj_time_series_weekly_auto.csv'
FILE_HS = 'output/hs_time_series_weekly_auto.csv'
# ==========================================

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def analyze_comprehensive_path(file_path, region_name):
    print(f"\n🚀 [종합 분석] {region_name} 토양-영양-생육-결과 연결고리 분석")

    if not os.path.exists(file_path):
        print(f"❌ 오류: 파일이 없습니다 ({file_path})")
        return

    df = pd.read_csv(file_path)

    # 1. 분석 변수 정의
    soil_cols = ['soil_pH', 'soil_EC', 'soil_OM', 'soil_AVSi', 'soil_Mg']
    leaf_cols = ['leaf_N1', 'leaf_N2']  # 엽분석 데이터
    drone_peak_cols = [c for c in df.columns if 'Peak_Val' in c]  # 드론 Peak 값
    result_cols = ['yield_weight', 'yield_protein']

    # 데이터셋에 존재하는 컬럼만 선택
    all_cols = soil_cols + leaf_cols + drone_peak_cols + result_cols
    valid_cols = [c for c in all_cols if c in df.columns]

    # 데이터프레임 필터링
    df_analysis = df[valid_cols].dropna()
    print(f"   -> 분석 대상 샘플 수: {len(df_analysis)}개")

    # -----------------------------------------------------------
    # [Step 1] 상관관계 히트맵 (전체 연결고리 파악)
    # -----------------------------------------------------------
    plt.figure(figsize=(12, 10))
    corr = df_analysis.corr()

    # 마스크(삼각형) 처리로 가독성 확보
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                vmin=-1, vmax=1, center=0, linewidths=.5)
    plt.title(f'[{region_name}] 토양-영양-생육-결과 종합 상관관계')
    plt.tight_layout()
    plt.savefig(f'output/comprehensive_heatmap_{region_name}.png')
    print(f"🖼️ [Step 1] 전체 상관관계 맵 저장 완료")

    # -----------------------------------------------------------
    # [Step 2] 핵심 가설 검증 시각화 (Scatter Plot with Regression)
    # -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 가설 1: 토양(OM) -> 엽질소(Leaf_N2)
    if 'soil_OM' in df.columns and 'leaf_N2' in df.columns:
        sns.regplot(data=df, x='soil_OM', y='leaf_N2', ax=axes[0, 0], color='brown')
        axes[0, 0].set_title(f'가설 1: 땅심(OM)이 좋으면 후기 영양(Leaf_N2)도 좋은가?')

    # 가설 2: 엽질소(Leaf_N2) -> 드론(NDRE Peak)
    # (드론 데이터 중 NDRE가 있으면 사용, 없으면 첫번째꺼)
    ndre_col = next((c for c in drone_peak_cols if 'NDRE' in c), drone_peak_cols[0] if drone_peak_cols else None)
    if 'leaf_N2' in df.columns and ndre_col:
        sns.regplot(data=df, x='leaf_N2', y=ndre_col, ax=axes[0, 1], color='green')
        axes[0, 1].set_title(f'가설 2: 엽질소(Leaf_N2)와 드론({ndre_col})은 일치하는가?')

    # 가설 3: 엽질소(Leaf_N2) -> 단백질(Protein)
    if 'leaf_N2' in df.columns and 'yield_protein' in df.columns:
        sns.regplot(data=df, x='leaf_N2', y='yield_protein', ax=axes[1, 0], color='purple')
        axes[1, 0].set_title(f'가설 3: 엽질소(Leaf_N2)가 단백질을 결정하는가?')

    # 가설 4: 규산(Si) -> 수확량(Yield)
    if 'soil_AVSi' in df.columns and 'yield_weight' in df.columns:
        # 2차 곡선 회귀 (과잉 구간 확인용)
        sns.regplot(data=df, x='soil_AVSi', y='yield_weight', ax=axes[1, 1], color='blue', order=2)
        axes[1, 1].set_title(f'가설 4: 규산(Si)은 수확량을 높이는가? (역U자형 검증)')

    plt.tight_layout()
    plt.savefig(f'output/theme3/comprehensive_hypothesis_{region_name}.png')
    print(f"🖼️ [Step 2] 가설 검증 그래프 저장 완료")

    # -----------------------------------------------------------
    # [Step 3] 최종 결정요인 중요도 분석 (Random Forest)
    # -----------------------------------------------------------
    # 목표: 단백질(Protein)을 결정짓는 1등 공신 찾기
    if 'yield_protein' in df.columns:
        X = df_analysis.drop(columns=result_cols)  # 결과 변수 제외하고 모두 설명변수로 사용
        y = df_analysis['yield_protein']

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # 중요도 추출
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

        # 시각화
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index, palette='magma')
        plt.title(f'[{region_name}] 단백질 함량 결정요인 중요도 (Top Factors)')
        plt.xlabel('영향력 (Importance Score)')
        plt.tight_layout()
        plt.savefig(f'output/theme3/feature_importance_protein_{region_name}.png')

        print(f"🖼️ [Step 3] 단백질 결정요인 순위 저장 완료")
        print(f"   👉 Top 3 요인: {importances.index[:3].tolist()}")

    # 결과 데이터 저장
    df_analysis.to_csv(f'output/theme3/comprehensive_data_{region_name}.csv', index=False, encoding='utf-8-sig')


# 실행
if __name__ == "__main__":
    analyze_comprehensive_path(FILE_GJ, "김제")
    analyze_comprehensive_path(FILE_HS, "화성")