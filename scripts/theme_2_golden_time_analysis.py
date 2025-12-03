"""
[테마 2] 생육 골든타임 분석 프로그램

목적: 시계열 식생지수 데이터를 분석하여 수확량 및 단백질 예측에 가장 중요한 시기(골든타임)를 파악

분석 방법:
- 각 시기별 식생지수와 수확량/단백질 간의 상관관계 분석
- 히트맵으로 시각화
- Top 3 중요 시기 도출

작성자: 농업 데이터 분석팀
버전: v1.0
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# ==========================================
# [설정] 파일 경로
# ==========================================
OUTPUT_DIR = 'output'
INPUT_FILES = {
    'Kimje': f'{OUTPUT_DIR}/gj_time_series_weekly_auto.csv',
    'Hwaseong': f'{OUTPUT_DIR}/hs_time_series_weekly_auto.csv'
}

# ==========================================
# [설정] 시각화 옵션
# ==========================================
# 한글 폰트 설정 (Windows: 'Malgun Gothic', Linux/Mac: 'DejaVu Sans')
# FONT_FAMILY = 'DejaVu Sans'  # Linux/Mac용 (한글 깨짐)
FONT_FAMILY = 'Malgun Gothic'  # Windows용 (한글 지원)

plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False

# 히트맵 색상 설정
HEATMAP_CMAP = 'coolwarm'
HEATMAP_DPI = 300


def validate_input_file(file_path):
    """입력 파일 검증"""
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        print(f"   현재 작업 디렉토리: {os.getcwd()}")
        return False
    return True


def load_data(file_path):
    """데이터 로드"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 데이터 로드 완료: {len(df)}개 샘플")
        return df
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None


def find_time_series_columns(df):
    """
    시계열 식생지수 컬럼 자동 탐지

    탐지 패턴:
    1. 숫자로 시작하는 컬럼 (예: 01_NDVI, 02_GNDVI, 15_LAI)
    2. 'Week_' 또는 'week_'로 시작하는 컬럼
    3. 날짜 형식 컬럼 (YYYY-MM-DD)
    """
    # 패턴 1: 숫자로 시작하고 '_'를 포함하는 컬럼
    pattern1_cols = [c for c in df.columns if c[0].isdigit() and '_' in c]

    # 패턴 2: Week로 시작하는 컬럼
    pattern2_cols = [c for c in df.columns if c.lower().startswith('week_')]

    # 패턴 3: 날짜 형식 (YYYY-MM-DD)
    pattern3_cols = []
    for c in df.columns:
        if '-' in c and len(c.split('-')) == 3:
            try:
                # 날짜 형식 검증
                parts = c.split('-')
                if len(parts[0]) == 4 and parts[0].isdigit():
                    pattern3_cols.append(c)
            except:
                pass

    # 모든 패턴 결합
    all_vi_cols = list(set(pattern1_cols + pattern2_cols + pattern3_cols))

    # 컬럼명 정렬 (시간 순서)
    all_vi_cols.sort()

    return all_vi_cols


def calculate_correlation(df, vi_cols, target_cols):
    """상관관계 계산"""
    # 타겟 변수 검증
    missing_targets = [t for t in target_cols if t not in df.columns]
    if missing_targets:
        print(f"⚠️ 경고: 타겟 변수가 없습니다: {missing_targets}")
        target_cols = [t for t in target_cols if t in df.columns]

    if not target_cols:
        print("❌ 오류: 분석할 타겟 변수가 없습니다.")
        return None

    # 상관관계 계산
    analysis_df = df[vi_cols + target_cols]
    corr_matrix = analysis_df.corr()

    # 타겟 변수와의 상관관계만 추출
    target_corr = corr_matrix.loc[vi_cols, target_cols]

    return target_corr


def create_heatmap(target_corr, region_name, output_dir):
    """히트맵 시각화"""
    plt.figure(figsize=(12, max(10, len(target_corr) * 0.3)))

    sns.heatmap(
        target_corr,
        annot=True,  # 값 표시
        cmap=HEATMAP_CMAP,  # 색상 맵
        fmt='.2f',  # 소수점 2자리
        center=0,  # 중심값 0
        linewidths=0.5,  # 셀 구분선
        cbar_kws={'label': 'Correlation'}
    )

    plt.title(f'{region_name} - Vegetation Index Time Series Correlation (Golden Time)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Target Variables', fontsize=12, fontweight='bold')
    plt.ylabel('Time Series Vegetation Indices', fontsize=12, fontweight='bold')
    plt.tight_layout()

    # 파일명에서 특수문자 제거
    safe_region_name = region_name.replace(' ', '_').replace('/', '-')
    output_path = f'{output_dir}/theme2_goldentime_{safe_region_name}.png'
    plt.savefig(output_path, dpi=HEATMAP_DPI, bbox_inches='tight')
    print(f"🖼️  히트맵 저장 완료: {output_path}")
    plt.close()


def print_top_indicators(target_corr, region_name, top_n=3):
    """Top N 중요 지표 출력"""
    print(f"\n{'=' * 70}")
    print(f"  [{region_name}] Golden Time Analysis Results")
    print(f"{'=' * 70}")

    for target in target_corr.columns:
        print(f"\n📊 Top {top_n} indicators for predicting '{target}':")
        print("-" * 70)

        top_indicators = target_corr[target].abs().sort_values(ascending=False).head(top_n)

        for i, (indicator, corr_value) in enumerate(top_indicators.items(), 1):
            actual_value = target_corr.loc[indicator, target]
            direction = "↑ Positive" if actual_value > 0 else "↓ Negative"
            bar = '█' * int(abs(corr_value) * 20)

            print(f"  {i}. {indicator:30s}: {actual_value:6.3f} {direction:12s} {bar}")

        print("-" * 70)


def save_correlation_results(target_corr, region_name, output_dir):
    """상관관계 결과 CSV 저장"""
    # 파일명에서 특수문자 제거
    safe_region_name = region_name.replace(' ', '_').replace('/', '-')
    output_path = f'{output_dir}/theme2_correlation_{safe_region_name}.csv'

    # 절대값 기준 내림차순 정렬 (각 타겟별로)
    sorted_corr = target_corr.copy()
    for col in sorted_corr.columns:
        sorted_corr = sorted_corr.sort_values(by=col, key=abs, ascending=False)

    sorted_corr.to_csv(output_path, encoding='utf-8-sig')
    print(f"💾 상관관계 CSV 저장 완료: {output_path}")


def analyze_golden_time(file_path, region_name):
    """
    생육 골든타임 분석 메인 함수

    Parameters:
    -----------
    file_path : str
        입력 CSV 파일 경로
    region_name : str
        지역 이름 (결과 파일명 및 그래프 제목에 사용)
    """
    print(f"\n{'=' * 70}")
    print(f"  [Theme 2] {region_name} - Growth Golden Time Analysis")
    print(f"{'=' * 70}")

    # 1. 입력 파일 검증
    if not validate_input_file(file_path):
        return

    # 2. 데이터 로드
    df = load_data(file_path)
    if df is None:
        return

    # 3. 시계열 식생지수 컬럼 탐지
    vi_cols = find_time_series_columns(df)

    if not vi_cols:
        print("⚠️  경고: 시계열 식생지수 컬럼을 찾을 수 없습니다.")
        print("   예상 형식: 01_NDVI, 02_GNDVI, Week_01_LAI, 2024-05-01_NDVI 등")
        print(f"   사용 가능한 컬럼: {df.columns.tolist()}")
        return

    print(f"\n📋 탐지된 시계열 컬럼: {len(vi_cols)}개")
    print(f"   첫 5개: {vi_cols[:5]}")
    if len(vi_cols) > 5:
        print(f"   마지막 5개: {vi_cols[-5:]}")

    # 4. 타겟 변수 설정
    target_cols = ['yield_weight', 'yield_protein']

    # 5. 상관관계 계산
    target_corr = calculate_correlation(df, vi_cols, target_cols)
    if target_corr is None:
        return

    # 6. 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 7. 히트맵 생성
    create_heatmap(target_corr, region_name, OUTPUT_DIR)

    # 8. Top 지표 출력
    print_top_indicators(target_corr, region_name, top_n=3)

    # 9. 결과 CSV 저장
    save_correlation_results(target_corr, region_name, OUTPUT_DIR)

    print(f"\n✅ [{region_name}] 분석 완료!")
    print(f"{'=' * 70}\n")


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("  [Theme 2] Growth Golden Time Analysis Program")
    print("=" * 70)
    print(f"📊 분석 목적: 시계열 식생지수 데이터로 골든타임 파악")
    print(f"📁 출력 디렉토리: {OUTPUT_DIR}/")
    print("=" * 70)

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 각 지역별 분석 실행
    for region_name, file_path in INPUT_FILES.items():
        analyze_golden_time(file_path, region_name)

    print("\n" + "=" * 70)
    print("  🎉 전체 분석 완료!")
    print("=" * 70)
    print(f"\n📁 생성된 파일:")
    print(f"   - 히트맵: {OUTPUT_DIR}/theme2_goldentime_*.png")
    print(f"   - CSV: {OUTPUT_DIR}/theme2_correlation_*.csv")


if __name__ == "__main__":
    main()
