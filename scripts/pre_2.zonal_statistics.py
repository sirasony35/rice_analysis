import os
import glob
import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats
import numpy as np

# ==========================================
# [설정] 경로 확인
# ==========================================
GEOJSON_FOLDER = '../geo_data/김제'
TIF_FOLDER = '../data/생육데이터/김제'
OUTPUT_FOLDER = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'gj_final_clean_mapped.csv')

# [필터링] 분석할 식생지수 (대소문자 무시)
TARGET_INDICES = ['NDVI', 'GNDVI', 'NDRE', 'OSAVI', 'LCI']


# ==========================================

def step1_clean_zonal_stats_mapped():
    print("\n🚀 [Step 1] 구역 통계 추출 (컬럼명 최적화 버전)")
    print("   (Sample Code 기준 매핑 + '회차_지수명' 컬럼 생성)")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    geojson_files = glob.glob(os.path.join(GEOJSON_FOLDER, '*.geojson'))
    tif_files = glob.glob(os.path.join(TIF_FOLDER, '*.tif'))

    if not geojson_files or not tif_files:
        print("❌ 오류: GeoJSON 또는 TIF 파일이 없습니다.")
        return

    all_result_dfs = []

    for geo_path in geojson_files:
        geo_name = os.path.basename(geo_path)
        print(f"\n📄 GeoJSON 로드 중: {geo_name}")

        gdf_original = gpd.read_file(geo_path)

        # 결과를 담을 딕셔너리
        extracted_data = {}
        count_processed = 0

        for tif_path in tif_files:
            tif_name = os.path.basename(tif_path)
            tif_name_no_ext = os.path.splitext(tif_name)[0]  # 확장자 제거

            # 1. 식생지수 파일인지 확인
            is_target = any(idx in tif_name.upper() for idx in TARGET_INDICES)
            if not is_target:
                continue

            # -------------------------------------------------
            # [핵심] 컬럼명 예쁘게 만들기 ('회차_지수명')
            # -------------------------------------------------
            # 예: 'GJR10_01_250619_NDVI' -> ['GJR10', '01', '250619', 'NDVI']
            try:
                parts = tif_name_no_ext.split('_')

                # 회차 (두 번째 조각)
                session = parts[1]

                # 지수명 (마지막 조각, 혹시 뒤에 숫자가 붙어있을 수 있으니 필터링)
                # parts 리스트 중에서 TARGET_INDICES에 있는 단어를 찾음
                index_name = next((part for part in parts if part.upper() in TARGET_INDICES), parts[-1])

                # 최종 컬럼명 생성 (예: 01_NDVI)
                final_col_name = f"{session}_{index_name.upper()}"

            except Exception:
                # 파싱 실패 시 파일명 그대로 사용 (안전장치)
                final_col_name = tif_name_no_ext
            # -------------------------------------------------

            print(f"   📸 처리: {tif_name} -> 컬럼명: [{final_col_name}] ...", end="")

            try:
                with rasterio.open(tif_path) as src:
                    # 데이터 읽기 및 클리닝 (이전과 동일)
                    data_array = src.read(1)
                    affine = src.transform
                    tif_crs = src.crs

                    if src.nodata is not None:
                        data_array = np.where(data_array == src.nodata, np.nan, data_array)
                    data_array = np.where(data_array < -9000, np.nan, data_array)  # 이상치 제거
                    data_array = np.where(data_array == 0, np.nan, data_array)  # 0 제거
                    data_array = np.where((data_array < -5) | (data_array > 5), np.nan, data_array)  # 범위 초과 제거

                    # 좌표계 맞추기
                    if gdf_original.crs != tif_crs:
                        gdf_working = gdf_original.to_crs(tif_crs)
                    else:
                        gdf_working = gdf_original

                    # 구역 통계 추출
                    stats = zonal_stats(
                        gdf_working,
                        data_array,
                        affine=affine,
                        stats="mean",
                        all_touched=True,
                        nodata=np.nan
                    )

                # 결과 저장 (새로운 컬럼명으로!)
                extracted_data[final_col_name] = [s['mean'] for s in stats]
                count_processed += 1
                print(" 완료.")

            except Exception as e:
                print(f" ❌ 실패: {e}")

        print(f"   -> 총 {count_processed}개의 데이터 처리 완료.")

        # 데이터 합치기
        if extracted_data:
            df_extracted = pd.DataFrame(extracted_data)
            # 원본 GeoJSON과 드론 데이터를 옆으로 붙임
            gdf_combined = pd.concat([gdf_original, df_extracted], axis=1)
            all_result_dfs.append(gdf_combined)

    # 최종 파일 저장
    if all_result_dfs:
        final_gdf = pd.concat(all_result_dfs, ignore_index=True)

        # Geometry 제거
        if 'geometry' in final_gdf.columns:
            df = pd.DataFrame(final_gdf.drop(columns='geometry'))
        else:
            df = pd.DataFrame(final_gdf)

        # -------------------------------------------------
        # [추가] 보기 좋게 컬럼 정렬 (sample_code 앞으로, 생육데이터 뒤로)
        # -------------------------------------------------
        # 1. 고정 컬럼 (기본 정보)
        fixed_cols = ['no', 'soil_code', 'sample_code', 'case', 'drying', 'analysis', 'addr',
                      'lat', 'lon', 'soil_pH', 'soil_EC', 'soil_OM', 'soil_AVP', 'soil_AVSi',
                      'soil_K', 'soil_Ca', 'soil_Mg', 'leaf_N1', 'leaf_N2', 'yield_weight',
                      'yield_moisture', 'yield_protein']

        # 2. 실제 존재하는 고정 컬럼만 선택
        existing_fixed = [c for c in fixed_cols if c in df.columns]

        # 3. 나머지 컬럼 (드론 데이터 등) - 이름순 정렬 (01_NDVI, 02_NDVI...)
        drone_cols = sorted([c for c in df.columns if c not in existing_fixed])

        # 4. 최종 순서 적용
        df = df[existing_fixed + drone_cols]
        # -------------------------------------------------

        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n✅ [성공] 깔끔하게 정리된 파일 저장 완료: {OUTPUT_FILE}")

        # 결과 미리보기
        print("\n--- 생성된 컬럼명 예시 ---")
        print(drone_cols[:5])

    else:
        print("\n⚠️ 저장할 데이터가 없습니다.")


if __name__ == "__main__":
    step1_clean_zonal_stats_mapped()