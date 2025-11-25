# -*- coding: utf-8 -*-
import os
import sys
import glob
import geopandas as gpd
import rasterio  # rasterio.Env를 사용하기 위해 import
from rasterstats import zonal_stats

# --- 1. 사용자 설정 부분 ---
GEOJSON_FOLDER = 'geo_json_data'
RASTER_FOLDER = 'drone_data'
OUTPUT_FOLDER = 'result_geojson'


# -------------------------

def main():
    """메인 실행 함수"""

    # === ★★★ 수정된 부분: 스크립트 실행 동안 GDAL 환경 설정 적용 ★★★ ===
    # GTIFF_SRS_SOURCE='EPSG' 설정으로 불필요한 CRS 경고 메시지를 제거합니다.
    with rasterio.Env(GTIFF_SRS_SOURCE='EPSG'):
        print("일괄 처리 스크립트 실행 시작...")

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            print(f"출력 폴더 생성: {OUTPUT_FOLDER}")

        geojson_files = glob.glob(os.path.join(GEOJSON_FOLDER, '*.geojson'))

        if not geojson_files:
            print(f"[오류] GeoJSON 입력 폴더에 파일이 없습니다: {GEOJSON_FOLDER}")
            return

        print(f"\n총 {len(geojson_files)}개의 GeoJSON 파일을 처리합니다.")

        for geojson_path in geojson_files:
            print(f"\n--- 처리 중인 파일: {os.path.basename(geojson_path)} ---")
            gdf = gpd.read_file(geojson_path)

            base_name = os.path.splitext(os.path.basename(geojson_path))[0]
            field_id = base_name.split('_')[-1].replace('-', '')
            print(f"필드명: {field_id}")

            raster_search_path = os.path.join(RASTER_FOLDER, f'{field_id}*.tif')
            raster_files = glob.glob(raster_search_path)

            if not raster_files:
                print(f"   [경고] '{field_id}'에 해당하는 래스터 파일을 찾을 수 없습니다. 건너<binary data, 2 bytes>니다.")
                continue

            print(f"   > 총 {len(raster_files)}개의 연관 래스터 파일을 찾았습니다. 구역 통계를 시작합니다.")
            original_crs = gdf.crs

            for raster_path in sorted(raster_files):
                raster_filename = os.path.basename(raster_path)
                try:
                    parts = raster_filename.split('_')
                    session = int(parts[1])
                    index_name = parts[3].split('.')[0]
                    column_name = f"{index_name}_{session}"

                    print(f"     - 계산 중: {raster_filename} -> '{column_name}' 컬럼")

                    with rasterio.open(raster_path) as src:
                        raster_crs = src.crs

                    if original_crs != raster_crs:
                        gdf_reprojected = gdf.to_crs(raster_crs)
                    else:
                        gdf_reprojected = gdf.copy()

                    stats = zonal_stats(gdf_reprojected, raster_path, stats="mean", all_touched=True)

                    mean_values = [s['mean'] if s['mean'] is not None else 0.0 for s in stats]
                    gdf[column_name] = mean_values

                except Exception as e:
                    print(f"     [오류] '{raster_filename}' 처리 중 문제 발생: {e}")

            base_name_with_ext = os.path.basename(geojson_path)
            name_part, extension = os.path.splitext(base_name_with_ext)
            new_output_filename = f"{name_part}_zonal_stats{extension}"
            output_path = os.path.join(OUTPUT_FOLDER, new_output_filename)

            gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')
            print(f"   [성공] 최종 결과 파일 저장 완료: {new_output_filename}")

        print("\n--- 모든 작업이 완료되었습니다. ---")


if __name__ == '__main__':
    main()