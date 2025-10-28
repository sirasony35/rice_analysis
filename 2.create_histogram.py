# -*- coding: utf-8 -*-
import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 1. 사용자 설정 부분 ---
INPUT_FOLDER = '../Drone/25년 생육데이터/벼/1.김제'  # 입력 폴더 경로
OUTPUT_FOLDER = '김제_histogram'  # 출력 폴더 경로
# -------------------------

# --- 처리 대상 식생 지수 목록 ---
# 이 목록에 포함된 이름이 파일명에 있어야 처리됩니다.
VALID_INDEX_NAMES = ['OSAVI', 'GNDVI', 'NDVI', 'LCI', 'NDRE']


# -------------------------

def create_raster_histogram(raster_path, output_path):
    """단일 래스터 파일의 히스토그램을 생성하고 통계를 출력합니다."""
    print(f"-> 처리 중: {os.path.basename(raster_path)}")
    try:
        with rasterio.open(raster_path) as src:
            image_data = src.read(1)
            nodata_value = src.nodata

        if nodata_value is not None:
            valid_data = image_data[image_data != nodata_value].flatten()
        else:
            valid_data = image_data.flatten()

        print(f"Original pixel count (incl. NoData if any): {image_data.size}")  # 원본 픽셀 수 추가

        valid_data = valid_data[(valid_data > -2) & (valid_data < 5)]
        print(f"   [정보] 분석할 총 픽셀 수: {valid_data.size}")

        if valid_data.size < 2:
            print("   [경고] 분석할 유효한 데이터가 부족합니다. 건너<binary data, 2 bytes><binary data, 2 bytes><binary data, 2 bytes>니다.")
            return

        counts, bin_edges = np.histogram(valid_data, bins=256)

        print("   [분석] 픽셀 수가 가장 많은 상위 5개 구간(Bin):")
        sorted_indices = np.argsort(counts)[::-1]

        print("   -----------------------------------------")
        print("   | 순위 |      구간 (Value)     | 픽셀 수 |")
        print("   -----------------------------------------")
        for i in range(5):
            if i < len(sorted_indices):
                index = sorted_indices[i]
                bin_start = bin_edges[index]
                bin_end = bin_edges[index + 1]
                count = counts[index]
                print(f"   |  {i + 1}   | {bin_start:.4f} - {bin_end:.4f} | {count:>7} |")
        print("   -----------------------------------------")

        peak1_index = sorted_indices[0]
        peak1_value = (bin_edges[peak1_index] + bin_edges[peak1_index + 1]) / 2
        peak1_count = counts[peak1_index]

        peak2_index = sorted_indices[1]
        peak2_value = (bin_edges[peak2_index] + bin_edges[peak2_index + 1]) / 2
        peak2_count = counts[peak2_index]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(valid_data, bins=256, color='skyblue', edgecolor='black')
        ax.axvline(peak1_value, color='red', linestyle='--', linewidth=2, label=f'1st Peak: {peak1_value:.4f}')
        ax.text(peak1_value, peak1_count, f' 1st Peak\n {peak1_value:.4f}', color='red', ha='left', va='bottom',
                fontsize=12, weight='bold')
        ax.axvline(peak2_value, color='purple', linestyle=':', linewidth=2, label=f'2nd Peak: {peak2_value:.4f}')
        ax.text(peak2_value, peak2_count, f' 2nd Peak\n {peak2_value:.4f}', color='purple', ha='right', va='bottom',
                fontsize=12, weight='bold')

        ax.set_title(f'{os.path.basename(raster_path)} - Pixel Value Distribution', fontsize=16)
        ax.set_xlabel('Vegetation Index Value', fontsize=12)
        ax.set_ylabel('Pixel Count (Frequency)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    except Exception as e:
        print(f"   [오류] 처리 중 문제가 발생했습니다: {e}")


# === ★★★ 수정된 main 함수 ★★★ ===
def main():
    """메인 실행 함수"""
    with rasterio.Env(GTIFF_SRS_SOURCE='EPSG'):
        print("히스토그램 일괄 생성 스크립트 실행 시작...")

        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("[경고] 'Malgun Gothic' 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            print(f"출력 폴더 생성: {OUTPUT_FOLDER}")

        raster_files = glob.glob(os.path.join(INPUT_FOLDER, '*.tif'))
        if not raster_files:
            print(f"[오류] 입력 폴더에 TIF 파일이 없습니다: {INPUT_FOLDER}")
            return

        print(f"\n총 {len(raster_files)}개의 TIF 파일을 확인합니다...")

        processed_count = 0  # 처리된 파일 수를 세기 위한 변수

        for raster_path in raster_files:
            filename_base = os.path.basename(raster_path)
            filename_upper = filename_base.upper()  # 대소문자 구분 없이 비교

            # --- 파일 이름에 유효한 식생 지수 이름이 포함되어 있는지 확인 ---
            is_valid_file = any(index_name in filename_upper for index_name in VALID_INDEX_NAMES)

            if not is_valid_file:
                print(
                    f"-> '{filename_base}' 파일 이름에 유효한 식생 지수가 없어 건너<binary data, 2 bytes><binary data, 2 bytes><binary data, 2 bytes>니다.")
                continue  # 다음 파일로 넘어감
            # --- 확인 로직 끝 ---

            # 유효한 파일만 히스토그램 생성
            base_name_no_ext = os.path.splitext(filename_base)[0]
            output_filename = f"{base_name_no_ext}_histogram.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            create_raster_histogram(raster_path, output_path)
            processed_count += 1  # 처리된 파일 수 증가

        print(f"\n--- 총 {processed_count}개의 유효한 파일에 대한 작업이 완료되었습니다. ---")
        print(f"결과물은 '{OUTPUT_FOLDER}' 폴더에 저장되었습니다.")


if __name__ == '__main__':
    main()