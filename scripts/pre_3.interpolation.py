import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
from datetime import datetime, timedelta

# ==========================================
# [설정] 입력 파일 및 TIF 폴더 경로 (반드시 확인!)
# ==========================================
# 1. Step 1에서 만든 CSV 파일 경로
INPUT_FILE = 'output/hs_final_matched.csv'

# 2. 날짜를 추출할 원본 TIF 폴더 (Step 1과 동일)
TIF_FOLDER = '../data/생육데이터/화성'

# 3. 결과 저장 경로
OUTPUT_FILE = 'output/hs_time_series_weekly_auto.csv'
OUTPUT_IMG_DIR = 'output/hs_growth_curves_weekly'

# 분석할 식생지수 목록
TARGET_INDICES = ['NDVI', 'GNDVI', 'NDRE', 'OSAVI', 'LCI']
# ==========================================

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def get_session_dates_from_tifs(folder_path):
    """
    TIF 파일명을 스캔하여 {회차: 날짜} 딕셔너리를 자동으로 생성하는 함수
    파일명 형식 예시: GJR1_01_250619_GNDVI.tif
    """
    print(f"\n📅 촬영일자 자동 추출 중... (폴더: {folder_path})")

    date_map = {}
    # 폴더 내 모든 tif 파일 검색
    tif_files = glob.glob(os.path.join(folder_path, "*.tif"))

    if not tif_files:
        print("❌ 오류: TIF 파일이 없습니다. 경로를 확인해주세요.")
        return None

    for f in tif_files:
        filename = os.path.basename(f)
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')

        # 파일명 형식이 맞는지 확인 (최소 3개 조각 이상)
        if len(parts) >= 3:
            session = parts[1]  # '01'
            date_str = parts[2]  # '250619'

            # 날짜 형식이 6자리 숫자인지 확인
            if len(date_str) == 6 and date_str.isdigit():
                try:
                    # 250619 -> 2025-06-19 변환
                    full_date = datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
                    date_map[session] = full_date
                except ValueError:
                    continue

                    # 회차순 정렬
    sorted_map = dict(sorted(date_map.items()))

    if not sorted_map:
        print("⚠️ 날짜 추출 실패: 파일명 형식을 확인해주세요 (예: *_01_250619_*.tif)")
        return None

    print(f"   ✅ 추출된 일정: {sorted_map}")
    return sorted_map


def step2_auto_interpolation_final():
    print("\n🚀 [Step 2] 자동 날짜 매핑 및 주 단위(Weekly) 시계열 분석 시작")

    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)

    # 1. 날짜 정보 자동 추출
    SESSION_DATES = get_session_dates_from_tifs(TIF_FOLDER)
    if not SESSION_DATES:
        return

    # 2. 데이터 로드
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 오류: 입력 파일({INPUT_FILE})이 없습니다.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"📄 데이터 로드: {len(df)}개 포인트")

    # 3. 날짜 처리 (X축: Day of Year)
    session_doy = {}
    # 기준 연도 추출 (첫 번째 날짜의 연도 사용)
    base_year = int(list(SESSION_DATES.values())[0][:4])

    for sess, date_str in SESSION_DATES.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        session_doy[sess] = dt.timetuple().tm_yday

    # 보간 X축 생성 (7일 간격)
    sorted_doys = sorted(session_doy.values())
    start_doy = sorted_doys[0]
    end_doy = sorted_doys[-1]

    # 시작일부터 끝일까지 7일 간격으로 생성
    x_new = np.arange(start_doy, end_doy + 1, 7)

    print(f"📊 분석 기간: DOY {start_doy} ~ {end_doy} (7일 간격, 총 {len(x_new)}개 포인트)")

    # 4. 지수별 보간 및 Peak 찾기
    for index_name in TARGET_INDICES:
        print(f"\n🔍 분석 중: {index_name} ...")

        # [핵심 수정] 컬럼 찾기 로직 강화
        # 기존: if index_name in c (NDVI가 GNDVI에도 포함되어 문제 발생)
        # 수정: c.endswith(f"_{index_name}") (정확히 해당 지수로 끝나는 컬럼만 선택)
        cols = [
            c for c in df.columns
            if c.endswith(f"_{index_name}") and c.split('_')[0] in SESSION_DATES
        ]
        cols = sorted(cols)

        if not cols:
            print(f"   ⚠️ 데이터 없음 (Skip: {index_name})")
            continue

        peak_values = []
        peak_dates = []

        count_success = 0
        for idx, row in df.iterrows():
            # 데이터 값 가져오기
            y_values = row[cols].values.astype(float)
            x_values = np.array([session_doy[c.split('_')[0]] for c in cols])

            # 결측치 체크 (데이터가 3개 미만이면 스플라인 불가)
            valid_mask = ~np.isnan(y_values)
            if np.sum(valid_mask) < 3:
                peak_values.append(np.nan)
                peak_dates.append(np.nan)
                continue

            try:
                # 유효한 데이터만 사용하여 보간
                x_valid = x_values[valid_mask]
                y_valid = y_values[valid_mask]

                # Cubic Spline 보간
                cs = CubicSpline(x_valid, y_valid, bc_type='natural')
                y_new = cs(x_new)

                # Peak 찾기
                max_idx = np.argmax(y_new)
                peak_val = y_new[max_idx]
                peak_doy = x_new[max_idx]

                # DOY -> 날짜(MM-DD) 변환
                peak_date_obj = datetime(base_year, 1, 1) + timedelta(days=int(peak_doy) - 1)
                peak_date_str = peak_date_obj.strftime("%m-%d")

                peak_values.append(peak_val)
                peak_dates.append(peak_date_str)
                count_success += 1

                # [시각화] 모든 포인트에 대해 그래프 저장 (제한 해제)
                if True:
                    fig, ax = plt.subplots(figsize=(10, 5))

                    # 관측 데이터 (점)
                    ax.plot(x_valid, y_valid, 'o', label='Observed (Monthly)', markersize=8, color='black')
                    # 보간 데이터 (선)
                    ax.plot(x_new, y_new, '-', label='Weekly Spline', color='green', alpha=0.7)
                    # Peak 지점 (별)
                    ax.plot(peak_doy, peak_val, 'r*', markersize=15, label=f'Peak: {peak_date_str}')

                    # X축 눈금 날짜로 변환
                    def doy_to_date_str(doy):
                        return (datetime(base_year, 1, 1) + timedelta(days=int(doy) - 1)).strftime("%m-%d")

                    # X축 틱 설정 (14일 간격)
                    xticks_doy = np.arange(x_new[0], x_new[-1], 14)
                    xticks_labels = [doy_to_date_str(d) for d in xticks_doy]

                    ax.set_xticks(xticks_doy)
                    ax.set_xticklabels(xticks_labels, rotation=45)

                    # 제목에 Sample Code 표시
                    sample_code = row.get('sample_code', f'Sample_{idx}')
                    ax.set_title(f"Growth Curve: {sample_code} ({index_name})")
                    ax.set_xlabel("Date")
                    ax.set_ylabel(index_name)
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                    plt.tight_layout()
                    # 파일명에 sample_code 포함
                    plt.savefig(os.path.join(OUTPUT_IMG_DIR, f"{sample_code}_{index_name}.png"))
                    plt.close()

            except Exception:
                peak_values.append(np.nan)
                peak_dates.append(np.nan)

        # 결과 저장: Peak 정보
        df[f'{index_name}_Peak_Val'] = peak_values
        df[f'{index_name}_Peak_Date'] = peak_dates
        print(f"   ✅ 처리 완료: {count_success} / {len(df)} 건")

    # 5. 최종 파일 저장
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n💾 [완료] 결과 저장됨: {OUTPUT_FILE}")
    print(f"   -> Peak 값(Val)과 날짜(Date) 컬럼이 추가되었습니다.")
    print(f"   -> 그래프 확인: {OUTPUT_IMG_DIR} (총 {len(df)}개 포인트)")


if __name__ == "__main__":
    step2_auto_interpolation_final()