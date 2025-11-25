# -*- coding: utf-8 -*-
import os
import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import shutil

# --- 1. 사용자 설정 부분 ---
INPUT_RASTER_FOLDER = '../data/생육데이터/화성'  ## 원본 데이터 저장 폴더
OUTPUT_RASTER_FOLDER = '../data/생육데이터/화성/hs_data_reprojected_5179'  ## 변환된 데이터 저장 폴더

TARGET_CRS_STRING = 'EPSG:5179'  ## 변환원하는 자표계


# -------------------------

def main():
    """메인 실행 함수"""
    print("래스터 좌표계 변환 스크립트 실행 시작...")

    if not os.path.exists(OUTPUT_RASTER_FOLDER):
        os.makedirs(OUTPUT_RASTER_FOLDER)
        print(f"출력 폴더 생성: {OUTPUT_RASTER_FOLDER}")

    raster_files = glob.glob(os.path.join(INPUT_RASTER_FOLDER, '*.tif'))

    if not raster_files:
        print(f"[오류] 입력 폴더에 TIF 파일이 없습니다: {INPUT_RASTER_FOLDER}")
        return

    print(f"\n총 {len(raster_files)}개의 파일을 확인합니다.")

    # 목표 CRS의 코드(숫자) 부분만 문자열로 추출
    target_epsg_code_str = TARGET_CRS_STRING.split(':')[-1]

    for raster_path in raster_files:
        filename = os.path.basename(raster_path)
        output_path = os.path.join(OUTPUT_RASTER_FOLDER, filename)

        with rasterio.open(raster_path) as src:
            source_crs = src.crs
            is_target_crs = False

            # === ★★★ 최종 수정된 부분: 단순하고 강력한 문자열 검색 ★★★ ===
            if source_crs:
                source_wkt = source_crs.to_wkt()
                # WKT 문자열 안에 'EPSG'와 목표 코드('5179')가 모두 있는지 확인
                if 'EPSG' in source_wkt and target_epsg_code_str in source_wkt:
                    is_target_crs = True

            print(f"-> 확인 중: {filename} (목표 CRS와 동일한가? {is_target_crs})")

            if not is_target_crs:
                print(f"   [변환 필요] 좌표계를 {TARGET_CRS_STRING}로 재투영합니다...")

                target_crs_object = CRS.from_string(TARGET_CRS_STRING)
                transform, width, height = calculate_default_transform(
                    source_crs, target_crs_object, src.width, src.height, *src.bounds)

                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs_object,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs_object,
                            resampling=Resampling.nearest)
                print(f"   [성공] 변환된 파일 저장 완료: {filename}")
            else:
                print("   [통과] 좌표계가 이미 올바릅니다. 파일을 건너뜁니다.")
                # shutil.copy(raster_path, output_path)

    print("\n--- 모든 래스터 파일 처리가 완료되었습니다. ---")


if __name__ == '__main__':
    main()