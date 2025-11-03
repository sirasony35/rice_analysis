# -*- coding: utf-8 -*-
import os
import glob
import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 1. 사용자 설정 부분 ---
GEOJSON_FOLDER = 'result_geojson'
TARGET_VARIABLES = ['yield', 'protein']


# -------------------------

def main():
    """메인 실행 함수"""
    print("상관관계 분석 스크립트 실행 시작...")

    # 1. 모든 GeoJSON 파일 목록 가져와서 하나의 DataFrame으로 합치기
    geojson_files = glob.glob(os.path.join(GEOJSON_FOLDER, '*_zonal_stats.geojson'))
    if not geojson_files:
        print(f"[오류] GeoJSON 결과 폴더에 파일이 없습니다: {GEOJSON_FOLDER}")
        return

    gdf_list = [gpd.read_file(f) for f in geojson_files]
    full_df = pd.concat(gdf_list, ignore_index=True)
    print(f"총 {len(geojson_files)}개 파일, {len(full_df)}개 레코드(구역)를 성공적으로 불러왔습니다.")

    # 2. 분석에 사용할 컬럼만 선택하기
    index_names = ['BNVI', 'NDVI', 'GNDVI', 'LCI', 'MTCI', 'NDRE']
    predictor_variables = [col for col in full_df.columns if col.split('_')[0] in index_names]
    analysis_columns = TARGET_VARIABLES + predictor_variables
    analysis_df = full_df[analysis_columns].copy()

    # 한글 폰트 설정
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("[경고] 'Malgun Gothic' 폰트를 찾을 수 없습니다. 히트맵의 한글이 깨질 수 있습니다.")

    # === ★★★ 변경된 부분: 피어슨과 스피어만 상관관계 및 히트맵 모두 생성 ★★★ ===
    correlation_methods = ['pearson', 'spearman']

    for method in correlation_methods:
        print(f"\n\n{'=' * 20} {method.capitalize()} 상관관계 분석 {'=' * 20}")

        # 3. 지정된 방법으로 상관관계 행렬 계산
        corr_matrix = analysis_df.corr(method=method)

        # 4. 텍스트 결과 출력
        for target in TARGET_VARIABLES:
            sorted_corr = corr_matrix[target].drop(TARGET_VARIABLES).abs().sort_values(ascending=False)
            print(f"\n--- '{target}'와(과)의 {method.capitalize()} 상관관계 상위 10개 ---")
            print(sorted_corr.head(10))

        # 5. 히트맵 시각화 및 저장
        print(f"\n--- {method.capitalize()} 상관관계 히트맵 생성 ---")
        heatmap_data = corr_matrix.loc[predictor_variables, TARGET_VARIABLES]

        plt.figure(figsize=(10, 14))
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title(f'수확량/단백질과 식생 지수의 {method.capitalize()} 상관관계', fontsize=16)
        plt.tight_layout()

        output_image_path = f'correlation_heatmap_{method}.png'
        plt.savefig(output_image_path, dpi=200)
        plt.close()  # 메모리 해제를 위해 그래프 닫기
        print(f"[성공] 히트맵이 '{output_image_path}' 파일로 저장되었습니다.")

    print("\n\n--- 모든 작업이 완료되었습니다. ---")


if __name__ == '__main__':
    main()