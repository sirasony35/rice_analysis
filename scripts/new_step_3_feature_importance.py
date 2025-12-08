import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# 1. 데이터 로드 및 전처리
gj_df = pd.read_csv('raw_data/gj_final_matched.csv')
hs_df = pd.read_csv('raw_data/hs_final_matched.csv')


# 드론 지수 컬럼 자동 식별 함수
def get_drone_cols(df):
    keywords = ['NDVI', 'GNDVI', 'NDRE', 'LCI', 'OSAVI']
    return [col for col in df.columns if any(k in col for k in keywords) and col[0].isdigit()]


# 변수 그룹 설정 (사용자 요청 반영)
# 결정요인: 토양 8종 + leaf_N2
determinants = ['soil_pH', 'soil_EC', 'soil_OM', 'soil_AVP', 'soil_AVSi',
                'soil_K', 'soil_Ca', 'soil_Mg', 'leaf_N1','leaf_N2']

# 예측요인: 드론 지수 (공통)
gj_drone = get_drone_cols(gj_df)
hs_drone = get_drone_cols(hs_df)
predictors = list(set(gj_drone) & set(hs_drone))
predictors.sort()

target_yield = 'yield_weight'
target_protein = 'yield_protein'


# 2. 분석 함수 정의
def analyze_separated_importance(df, region_name):
    # 전처리: 수확량 0.1 미만 제거
    df_clean = df[df[target_yield] >= 0.1].copy()

    # 결측치 처리 (평균 대치)
    imputer = SimpleImputer(strategy='mean')

    # 데이터 준비
    X_det = pd.DataFrame(imputer.fit_transform(df_clean[determinants]), columns=determinants)
    X_pred = pd.DataFrame(imputer.fit_transform(df_clean[predictors]), columns=predictors)
    y_yield = df_clean[target_yield]
    y_prot = df_clean[target_protein]

    # 모델링 및 중요도 추출 함수
    def get_importance(X, y, feature_type, target_name):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        imp['Type'] = feature_type
        imp['Target'] = target_name
        imp['Region'] = region_name
        return imp

    # 1. 수확량 분석
    imp_yield_det = get_importance(X_det, y_yield, 'Determinant', 'Yield')
    imp_yield_pred = get_importance(X_pred, y_yield, 'Predictor', 'Yield')

    # 2. 단백질 분석
    imp_prot_det = get_importance(X_det, y_prot, 'Determinant', 'Protein')
    imp_prot_pred = get_importance(X_pred, y_prot, 'Predictor', 'Protein')

    return imp_yield_det, imp_yield_pred, imp_prot_det, imp_prot_pred


# 3. 실행
gj_y_det, gj_y_pred, gj_p_det, gj_p_pred = analyze_separated_importance(gj_df, 'Gimje')
hs_y_det, hs_y_pred, hs_p_det, hs_p_pred = analyze_separated_importance(hs_df, 'Hwaseong')

# 4. 시각화
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.4)


# Helper function for plotting
def plot_imp(ax, data, title, color):
    sns.barplot(x='Importance', y='Feature', data=data.head(10), ax=ax, palette=color)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')


# Row 1: Yield (수확량)
plot_imp(axes[0, 0], gj_y_det, 'Gimje: Yield Determinants (Soil+Leaf)', 'Greens_r')
plot_imp(axes[0, 1], gj_y_pred, 'Gimje: Yield Predictors (Drone)', 'Blues_r')
plot_imp(axes[0, 2], hs_y_det, 'Hwaseong: Yield Determinants (Soil+Leaf)', 'Oranges_r')
plot_imp(axes[0, 3], hs_y_pred, 'Hwaseong: Yield Predictors (Drone)', 'Reds_r')

# Row 2: Protein (단백질)
plot_imp(axes[1, 0], gj_p_det, 'Gimje: Protein Determinants (Soil+Leaf)', 'Greens_r')
plot_imp(axes[1, 1], gj_p_pred, 'Gimje: Protein Predictors (Drone)', 'Blues_r')
plot_imp(axes[1, 2], hs_p_det, 'Hwaseong: Protein Determinants (Soil+Leaf)', 'Oranges_r')
plot_imp(axes[1, 3], hs_p_pred, 'Hwaseong: Protein Predictors (Drone)', 'Reds_r')

plt.suptitle('Step 3. Analysis of Determinants vs Predictors (Regional)', fontsize=18)
plt.savefig('output/new_step/step3_separated_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 결과 텍스트 출력
print("=== [Gimje] Analysis Results ===")
print("[Yield Determinants]\n", gj_y_det.head(3))
print("[Yield Predictors]\n", gj_y_pred.head(3))
print("[Protein Determinants]\n", gj_p_det.head(3))
print("[Protein Predictors]\n", gj_p_pred.head(3))

print("\n=== [Hwaseong] Analysis Results ===")
print("[Yield Determinants]\n", hs_y_det.head(3))
print("[Yield Predictors]\n", hs_y_pred.head(3))
print("[Protein Determinants]\n", hs_p_det.head(3))
print("[Protein Predictors]\n", hs_p_pred.head(3))