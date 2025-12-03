import pandas as pd

# Load Summary Files (업로드된 파일 사용)
df_gj = pd.read_csv('output/theme1/theme1_gj_quadrant_summary.csv')
df_hs = pd.read_csv('output/theme1/theme1_hs_quadrant_summary.csv')


def create_final_report_csv(df, region_name):
    # 진단 로직 매핑
    def get_diagnosis(group_name):
        if 'Target 1' in group_name or 'Q4' in group_name:
            return '[성공] 최적 모델 (Best)'
        elif 'Q1' in group_name:
            return '[위험] 과비/고단백 (Over)'
        elif 'Q2' in group_name:
            return '[불량] 저수확/고단백 (Bad)'
        elif 'Q3' in group_name:
            return '[부족] 영양 결핍 (Lack)'
        elif 'Q5' in group_name:
            return '[보통] 잠재력 보유 (Normal)'
        else:
            return '기타'

    df['Region'] = region_name
    df['Diagnosis'] = df['Group'].apply(get_diagnosis)

    # 주요 토양 성분 포함하여 정리
    cols_order = ['Region', 'Group', '비율(%)', 'yield_weight', 'yield_protein', 'Diagnosis',
                  'soil_OM', 'soil_AVSi', 'soil_Mg', 'soil_pH']

    # 컬럼명 한글화 (보고서용)
    rename_dict = {
        'yield_weight': '평균 수확량(kg)',
        'yield_protein': '평균 단백질(%)',
        'soil_OM': '유기물(OM)',
        'soil_AVSi': '유효규산(Si)',
        'soil_Mg': '마그네슘(Mg)',
        'soil_pH': '산도(pH)'
    }

    result = df[cols_order].rename(columns=rename_dict).round(2)
    return result


# 데이터 생성 및 병합
report_gj = create_final_report_csv(df_gj, "김제")
report_hs = create_final_report_csv(df_hs, "화성")
final_report = pd.concat([report_gj, report_hs], ignore_index=True)

# 저장 및 출력
output_filename = 'final_regional_analysis_summary.csv'
final_report.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\n✅ 최종 보고서용 CSV 저장 완료: {output_filename}")
print("\n--- [최종 데이터 미리보기] ---")
print(final_report.to_markdown(index=False))