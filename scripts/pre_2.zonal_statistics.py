import os
import glob
import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats
import numpy as np
import numpy.ma as ma
import re

# ==========================================
# [설정] 경로를 수정해주세요
# ==========================================
GEOJSON_FOLDER = '../geo_data/화성'
TIF_FOLDER = '../data/생육데이터/화성'
OUTPUT_FOLDER = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'hs_final_matched.csv')

# [필터링] 분석할 식생지수 (대소문자 무시)
TARGET_INDICES = ['NDVI', 'GNDVI', 'NDRE', 'OSAVI', 'LCI']


# ==========================================

def step1_smart_matching_stats():
    print("\n🚀 [Step 1] 필지별 스마트 매칭 구역 통계 시작")
    print("   (TIF 파일명 'GJR1' <-> GeoJSON 'GJ-R1' 자동 매핑)")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 1. GeoJSON 파일 찾기 (하나만 있다고 가정하거나 첫번째 것 사용)
    geojson_files = glob.glob(os.path.join(GEOJSON_FOLDER, '*.geojson'))
    if not geojson_files:
        print("❌ 오류: GeoJSON 파일이 없습니다.")
        return

    geo_path = geojson_files[0]  # 첫 번째 파일 사용
    print(f"\n📄 기준 GeoJSON 로드: {os.path.basename(geo_path)}")

    # GeoJSON 로드 및 인덱스 설정 (나중에 값 업데이트를 위해 중요)
    gdf_master = gpd.read_file(geo_path)

    # 결과 저장을 위한 DataFrame 복사 (Geometry 제외)
    df_result = pd.DataFrame(gdf_master.drop(columns='geometry'))

    # 2. TIF 파일 목록 가져오기
    tif_files = glob.glob(os.path.join(TIF_FOLDER, '*.tif'))
    if not tif_files:
        print("❌ 오류: TIF 파일이 없습니다.")
        return

    print(f"   -> 총 {len(tif_files)}개의 TIF 파일을 분석합니다.\n")

    # 3. TIF 파일별 반복 처리
    for tif_path in tif_files:
        tif_name = os.path.basename(tif_path)
        tif_name_no_ext = os.path.splitext(tif_name)[0]

        # --- [파일명 파싱 로직] ---
        # 예: "GJR1_01_250619_NDVI"
        parts = tif_name_no_ext.split('_')

        if len(parts) < 4:
            print(f"⚠️ 스킵: 파일명 형식이 맞지 않음 ({tif_name})")
            continue

        parcel_id_tif = parts[0]  # GJR1
        session = parts[1]  # 01
        # 식생지수 찾기
        index_name = next((p for p in parts if p.upper() in TARGET_INDICES), None)

        if not index_name:
            continue  # 대상 지수가 아니면 스킵

        # 컬럼명 생성 (예: 01_NDVI)
        col_name = f"{session}_{index_name.upper()}"

        # --- [핵심: 필지 매칭 로직] ---
        # TIF의 'GJR1'을 GeoJSON의 'GJ-R1' 형태로 변환
        # 정규표현식으로 영문자(GJR)와 숫자(1)를 분리
        match = re.match(r"([a-zA-Z]+)(\d+)", parcel_id_tif)
        if match:
            prefix = match.group(1)  # GJR
            number = match.group(2)  # 1

            # 변환 규칙 적용: GJR -> GJ-R
            if prefix.upper() == 'GJR':
                target_prefix = 'GJ-R'
            elif prefix.upper() == 'HSR':  # 화성(HSR)인 경우 대비
                target_prefix = 'HS-R'
            else:
                target_prefix = prefix  # 모르면 그대로

            target_sample_code_start = f"{target_prefix}{number}"  # 예: GJ-R1
        else:
            # 패턴 매칭 실패 시 파일명 그대로 검색 시도
            target_sample_code_start = parcel_id_tif

        # GeoJSON에서 해당 필지에 속하는 포인트만 필터링
        # sample_code 컬럼에서 'GJ-R1'로 시작하거나 포함된 행 찾기
        target_indices = gdf_master[gdf_master['sample_code'].str.contains(target_sample_code_start, case=False)].index

        if len(target_indices) == 0:
            print(f"   pass: {tif_name} (매칭되는 포인트 없음: {target_sample_code_start})")
            continue

        # ---------------------------

        # 해당 컬럼이 결과 DF에 없으면 생성 (NaN으로 초기화)
        if col_name not in df_result.columns:
            df_result[col_name] = np.nan

        print(f"   📸 처리: {tif_name} -> 대상: {target_sample_code_start} ({len(target_indices)}개 포인트)")

        try:
            with rasterio.open(tif_path) as src:
                # 필터링된 포인트들의 Geometry만 가져오기
                target_gdf = gdf_master.loc[target_indices]

                # 좌표계 매칭
                if target_gdf.crs != src.crs:
                    target_gdf = target_gdf.to_crs(src.crs)

                # 데이터 읽기 및 마스킹 (이상한 값 제거)
                data = src.read(1)
                affine = src.transform

                # 마스킹: NoData, 0, 비정상 범위 제거
                mask_condition = (data < -5) | (data > 5) | (data == 0)
                masked_data = ma.masked_where(mask_condition, data)

                # 구역 통계 추출
                stats = zonal_stats(
                    target_gdf,
                    masked_data,
                    affine=affine,
                    stats="mean",
                    all_touched=True
                )

                # 추출된 값을 결과 DataFrame의 해당 인덱스에 업데이트
                # stats 순서와 target_indices 순서는 동일함
                values = [s['mean'] for s in stats]
                df_result.loc[target_indices, col_name] = values

        except Exception as e:
            print(f"     ❌ 오류 발생: {e}")

    # 4. 결과 저장 및 정렬
    # 컬럼 정렬: 기본정보 -> 토양 -> 드론(이름순)
    base_cols = ['no', 'soil_code', 'sample_code', 'addr', 'lat', 'lon']
    existing_base = [c for c in base_cols if c in df_result.columns]

    # 나머지 컬럼들
    other_cols = [c for c in df_result.columns if c not in existing_base]
    drone_cols = sorted([c for c in other_cols if c[0].isdigit()])  # 01_NDVI 등
    soil_cols = [c for c in other_cols if c not in drone_cols]

    final_cols = existing_base + soil_cols + drone_cols
    df_result = df_result[final_cols]

    df_result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ [성공] 매칭 및 병합 완료: {OUTPUT_FILE}")

    # 데이터 확인
    if drone_cols:
        print("\n--- 데이터 채워진 현황 (상위 5행) ---")
        print(df_result[['sample_code'] + drone_cols[:3]].head())


if __name__ == "__main__":
    step1_smart_matching_stats()