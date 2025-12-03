"""
수확량-단백질 트레이드오프 분석 프로그램 (최종 정리판)

분류 방법:
- 수확량: 사분위수 기준 (하위 25%, 상위 25%)
- 단백질: 고정값 6.0% 기준

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ==========================================
# [설정] 파일 경로
# ==========================================
INPUT_FILE = 'output/hs_time_series_weekly_auto.csv'
OUTPUT_CSV = 'output/theme1/theme1_hs_quadrant_summary.csv'
OUTPUT_IMG = 'output/theme1/theme1_hs_tradeoff_scatter.png'

# ==========================================
# [설정] 분류 기준
# ==========================================
YIELD_LOWER_PERCENTILE = 25  # 수확량 하위 25%
YIELD_UPPER_PERCENTILE = 75  # 수확량 상위 25%
PROTEIN_THRESHOLD = 6.0  # 단백질 고정 기준 (%)

# ==========================================
# [설정] 시각화 옵션
# ==========================================
SHOW_SAMPLE_ID = True  # 샘플 ID 표시 여부
SAMPLE_ID_FONTSIZE = 7  # 샘플 ID 폰트 크기
SAMPLE_ID_GROUPS = []  # 특정 그룹만 표시 (빈 리스트면 전체 표시)
# 예: ['Q1 (고수확/고단백)', 'Q3 (저수확/저단백)']
# ==========================================

# 한글 폰트 설정 (Linux 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def validate_input_file():
    """입력 파일 검증"""
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 오류: 입력 파일이 없습니다 ({INPUT_FILE})")
        print(f"   현재 작업 디렉토리: {os.getcwd()}")
        return False
    return True


def load_and_validate_data():
    """데이터 로드 및 검증"""
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ 데이터 로드 완료: {len(df)}개 샘플")

    # 필수 컬럼 확인
    required_cols = ['yield_weight', 'yield_protein']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ 오류: 필수 컬럼이 없습니다: {missing_cols}")
        return None

    # 단백질 데이터 검증
    if df['yield_protein'].sum() == 0:
        print("⚠️  경고: 단백질 데이터가 없습니다. (가상 데이터로 시뮬레이션합니다)")
        np.random.seed(42)
        df['yield_protein'] = 6.2 - (df['yield_weight'] - df['yield_weight'].mean()) * 0.3 + \
                              np.random.normal(0, 0.4, len(df))

    return df


def calculate_thresholds(df):
    """분류 기준값 계산"""
    yield_low = df['yield_weight'].quantile(YIELD_LOWER_PERCENTILE / 100)
    yield_high = df['yield_weight'].quantile(YIELD_UPPER_PERCENTILE / 100)
    protein_th = PROTEIN_THRESHOLD

    print(f"\n📏 계산된 기준값:")
    print(f"   수확량 하위 {YIELD_LOWER_PERCENTILE}%: {yield_low:.2f} kg")
    print(f"   수확량 상위 {100 - YIELD_UPPER_PERCENTILE}%: {yield_high:.2f} kg")
    print(f"   단백질 기준: {protein_th:.2f} %")
    print()

    return yield_low, yield_high, protein_th


def classify_samples(df, yield_low, yield_high, protein_th):
    """샘플 분류"""

    def classify(row):
        y, p = row['yield_weight'], row['yield_protein']

        if y >= yield_high and p >= protein_th:
            return 'Q1 (고수확/고단백)'
        elif y <= yield_low and p >= protein_th:
            return 'Q2 (저수확/고단백)'
        elif y <= yield_low and p < protein_th:
            return 'Q3 (저수확/저단백)'
        elif y >= yield_high and p < protein_th:
            return 'Q4 (고수확/저단백)'
        else:
            return 'Q5 (중간영역)'

    df['Group'] = df.apply(classify, axis=1)
    return df


def print_classification_results(df):
    """분류 결과 출력"""
    print("📊 그룹별 분류 결과:")
    print("-" * 70)

    group_counts = df['Group'].value_counts().sort_index()
    total_count = len(df)

    for group, count in group_counts.items():
        percentage = (count / total_count * 100)
        bar = '█' * int(percentage / 2)
        print(f"   {group:20s}: {count:3d}개 ({percentage:5.1f}%) {bar}")

    print("-" * 70)
    print()

    return group_counts


def create_scatter_plot(df, yield_low, yield_high, protein_th):
    """산점도 생성"""
    plt.figure(figsize=(16, 10))

    # 색상 팔레트
    color_palette = {
        'Q1 (고수확/고단백)': '#2ecc71',  # 초록
        'Q2 (저수확/고단백)': '#3498db',  # 파랑
        'Q3 (저수확/저단백)': '#e74c3c',  # 빨강
        'Q4 (고수확/저단백)': '#f39c12',  # 주황
        'Q5 (중간영역)': '#95a5a6'  # 회색
    }

    # 산점도 그리기
    for group in df['Group'].unique():
        group_data = df[df['Group'] == group]
        plt.scatter(
            group_data['yield_weight'],
            group_data['yield_protein'],
            label=group,
            color=color_palette.get(group, '#000000'),
            s=120,
            alpha=0.6,
            edgecolors='white',
            linewidth=0.5
        )

    # 샘플 ID 표시 (설정에 따라)
    if SHOW_SAMPLE_ID:
        # sample_id 컬럼 확인
        id_col = None
        for col in ['sample_id', 'sample_code', 'id', 'code', 'Sample_ID']:
            if col in df.columns:
                id_col = col
                break

        if id_col:
            # 표시할 데이터 필터링
            if SAMPLE_ID_GROUPS:
                df_to_show = df[df['Group'].isin(SAMPLE_ID_GROUPS)]
                print(f"ℹ️  샘플 ID 표시: {', '.join(SAMPLE_ID_GROUPS)} 그룹만 표시")
            else:
                df_to_show = df

            for idx, row in df_to_show.iterrows():
                # sample_id를 문자열로 변환 (정수든 문자열이든 처리 가능)
                sample_label = str(row[id_col])

                plt.annotate(
                    sample_label,
                    xy=(row['yield_weight'], row['yield_protein']),
                    xytext=(3, 3),  # 점에서 약간 떨어진 위치
                    textcoords='offset points',
                    fontsize=SAMPLE_ID_FONTSIZE,
                    alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='none', alpha=0.5)
                )
        else:
            print("⚠️  경고: sample_id 컬럼을 찾을 수 없습니다. ID 표시를 건너뜁니다.")

    # 기준선 표시
    plt.axvline(x=yield_low, color='red', linestyle='--', linewidth=2,
                alpha=0.5, label=f'Yield Lower {YIELD_LOWER_PERCENTILE}%: {yield_low:.1f}')
    plt.axvline(x=yield_high, color='red', linestyle='--', linewidth=2,
                alpha=0.5, label=f'Yield Upper {100 - YIELD_UPPER_PERCENTILE}%: {yield_high:.1f}')
    plt.axhline(y=protein_th, color='blue', linestyle='--', linewidth=2,
                alpha=0.5, label=f'Protein Threshold: {protein_th:.1f}%')

    # 그래프 설정
    plt.title('Yield-Protein Trade-off Analysis\n(Yield: Quartile, Protein: Fixed 6.0%)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Yield (kg/10a)', fontsize=13, fontweight='bold')
    plt.ylabel('Protein Content (%)', fontsize=13, fontweight='bold')

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11,
               frameon=True, shadow=True, fancybox=True)
    plt.grid(True, alpha=0.2, linestyle=':')
    plt.tight_layout()

    # 저장
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"🖼️  그래프 저장 완료: {OUTPUT_IMG}")
    plt.close()


def create_summary_table(df, group_counts):
    """요약 테이블 생성"""
    # 비교할 컬럼 선택
    cols_to_compare = ['soil_pH', 'soil_OM', 'soil_AVSi', 'soil_Mg'] + \
                      [c for c in df.columns if 'Peak_Val' in c] + \
                      ['yield_weight', 'yield_protein']

    valid_cols = [c for c in cols_to_compare if c in df.columns]

    # 그룹별 평균값 계산
    summary = df.groupby('Group')[valid_cols].mean(numeric_only=True)

    # 그룹별 개수 및 비율 계산
    total_count = len(df)
    group_stats = pd.DataFrame({
        '데이터_개수': group_counts,
        '비율(%)': (group_counts / total_count * 100).round(2)
    })

    # 개수와 비율을 summary에 추가
    summary = summary.join(group_stats)

    # 컬럼 순서 조정
    cols_ordered = ['데이터_개수', '비율(%)'] + \
                   [c for c in summary.columns if c not in ['데이터_개수', '비율(%)']]
    summary = summary[cols_ordered]

    # CSV 저장
    summary.to_csv(OUTPUT_CSV, encoding='utf-8-sig')
    print(f"💾 결과 CSV 저장 완료: {OUTPUT_CSV}")
    print()

    return summary


def print_summary_results(summary):
    """결과 요약 출력"""
    print("=" * 70)
    print("  [그룹별 평균 특성 비교]")
    print("=" * 70)

    # 주요 지표만 출력
    display_cols = ['데이터_개수', '비율(%)', 'yield_weight', 'yield_protein',
                    'soil_pH', 'soil_OM', 'soil_AVSi', 'soil_Mg']
    display_cols = [c for c in display_cols if c in summary.columns]

    print(summary[display_cols].round(2))
    print("=" * 70)
    print()


def print_insights(summary):
    """주요 인사이트 출력"""
    print("🔍 주요 인사이트:")
    print("-" * 70)

    # Q1 그룹 특징
    if 'Q1 (고수확/고단백)' in summary.index:
        q1_data = summary.loc['Q1 (고수확/고단백)']
        print(f"✓ Q1 (고수확/고단백): {q1_data['데이터_개수']:.0f}개 ({q1_data['비율(%)']:.1f}%)")
        print(f"  - 평균 수확량: {q1_data['yield_weight']:.2f} kg")
        print(f"  - 평균 단백질: {q1_data['yield_protein']:.2f}%")
        if 'soil_OM' in q1_data.index:
            print(f"  - 평균 유기물: {q1_data['soil_OM']:.2f}%")

    # Q3 그룹 특징
    if 'Q3 (저수확/저단백)' in summary.index:
        q3_data = summary.loc['Q3 (저수확/저단백)']
        print(f"✓ Q3 (저수확/저단백): {q3_data['데이터_개수']:.0f}개 ({q3_data['비율(%)']:.1f}%)")
        print(f"  - 평균 수확량: {q3_data['yield_weight']:.2f} kg")
        print(f"  - 평균 단백질: {q3_data['yield_protein']:.2f}%")
        if 'soil_OM' in q3_data.index:
            print(f"  - 평균 유기물: {q3_data['soil_OM']:.2f}%")

    # 중간 영역
    if 'Q5 (중간영역)' in summary.index:
        q5_data = summary.loc['Q5 (중간영역)']
        print(f"✓ Q5 (중간영역): {q5_data['데이터_개수']:.0f}개 ({q5_data['비율(%)']:.1f}%)")
        print(f"  - 수확량/단백질이 모두 중간 수준인 샘플")

    print("-" * 70)
    print()


def print_methodology():
    """분류 방법 설명"""
    print("=" * 70)
    print("  [분류 방법 설명]")
    print("=" * 70)
    print("📌 하이브리드 분류 방식:")
    print(f"   ✓ 수확량: 사분위수 기준")
    print(f"      - 하위 {YIELD_LOWER_PERCENTILE}% (저수확)")
    print(f"      - 상위 {100 - YIELD_UPPER_PERCENTILE}% (고수확)")
    print(f"      - 중간 영역 (나머지)")
    print(f"   ✓ 단백질: 고정값 {PROTEIN_THRESHOLD}% 기준")
    print(f"      - 고단백: {PROTEIN_THRESHOLD}% 이상")
    print(f"      - 저단백: {PROTEIN_THRESHOLD}% 미만")
    print()
    print("📊 장점:")
    print("   - 수확량: 극단값 영향 최소화, 상위/하위 명확 구분")
    print("   - 단백질: 농업적 목표값(6.0%) 기준, 해석 용이")
    print("   - 하이브리드: 통계적 안정성 + 도메인 지식 결합")
    print("=" * 70)


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("  [테마 1] 수확량-단백질 트레이드오프 분석")
    print("=" * 70)
    print(f"📊 분류 기준:")
    print(f"   - 수확량: 사분위수 (하위 {YIELD_LOWER_PERCENTILE}%, 상위 {100 - YIELD_UPPER_PERCENTILE}%)")
    print(f"   - 단백질: 고정값 {PROTEIN_THRESHOLD}%")
    print("=" * 70)
    print()

    # 1. 입력 파일 검증
    if not validate_input_file():
        return

    # 2. 데이터 로드
    df = load_and_validate_data()
    if df is None:
        return

    # 3. 기준값 계산
    yield_low, yield_high, protein_th = calculate_thresholds(df)

    # 4. 샘플 분류
    df = classify_samples(df, yield_low, yield_high, protein_th)

    # 5. 분류 결과 출력
    group_counts = print_classification_results(df)

    # 6. 산점도 생성
    create_scatter_plot(df, yield_low, yield_high, protein_th)

    # 7. 요약 테이블 생성
    summary = create_summary_table(df, group_counts)

    # 8. 결과 출력
    print_summary_results(summary)
    print_insights(summary)
    print_methodology()

    print("\n✅ 분석 완료!")


if __name__ == "__main__":
    main()
