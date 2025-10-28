# -*- coding: utf-8 -*-
import os
import sys
import glob

# --- 1. 사용자 설정 부분 ---
QGIS_INSTALL_PATH = 'C:/Program Files/QGIS 3.40.11'  # 예시 경로
INPUT_FOLDER = '../Drone/25년 생육데이터/벼/1.김제'
OUTPUT_FOLDER = '김제_vi'
OUTPUT_WIDTH_PX = 1200
# -------------------------

# --- 식생 지수별 등급/색상/라벨 규칙집 ---
### 밀
# # (경계 값, 색상 HEX 코드, 범례 라벨) 형식으로 각 규칙을 정의합니다.
# CLASSIFICATION_MAP = {
#     'BNVI': [
#         (0.20, '#c51f1e', '<= 0.20'),
#         (0.40, '#f5a361', '0.21 - 0.40'),
#         (0.55, '#faf7be', '0.41 - 0.55'),
#         (0.70, '#a1d193', '0.56 - 0.70'),
#         ('max', '#447cb9', '> 0.70')
#     ],
#     # ★★★ GNDVI를 NDVI보다 위로 이동 ★★★
#     'GNDVI': [
#         (0.30, '#c51f1e', '<= 0.30'),
#         (0.50, '#f5a361', '0.31 - 0.50'),
#         (0.65, '#faf7be', '0.51 - 0.65'),
#         (0.80, '#a1d193', '0.66 - 0.80'),
#         ('max', '#447cb9', '> 0.80')
#     ],
#     'NDVI': [
#         (0.25, '#c51f1e', '<= 0.25'),
#         (0.45, '#f5a361', '0.26 - 0.45'),
#         (0.60, '#faf7be', '0.46 - 0.60'),
#         (0.75, '#a1d193', '0.61 - 0.75'),
#         ('max', '#447cb9', '> 0.75')
#     ],
#     'LCI': [
#         (0.20, '#c51f1e', '<= 0.20'),
#         (0.40, '#f5a361', '0.21 - 0.40'),
#         (0.60, '#faf7be', '0.41 - 0.60'),
#         (0.85, '#a1d193', '0.61 - 0.85'),
#         ('max', '#447cb9', '> 0.85')
#     ],
#     'MTCI': [
#         (0.40, '#c51f1e', '<= 0.40'),
#         (0.80, '#f5a361', '0.41 - 0.80'),
#         (1.20, '#faf7be', '0.81 - 1.20'),
#         (1.80, '#a1d193', '1.21 - 1.80'),
#         ('max', '#447cb9', '> 1.80')
#     ],
#     'NDRE': [
#         (0.15, '#c51f1e', '<= 0.15'),
#         (0.30, '#f5a361', '0.16 - 0.30'),
#         (0.45, '#faf7be', '0.31 - 0.45'),
#         (0.60, '#a1d193', '0.46 - 0.60'),
#         ('max', '#447cb9', '> 0.60')
#     ]
# }

### 벼
## 각 등급별 이상 ~ 미만
CLASSIFICATION_MAP = {
    'OSAVI': [
        (0.25, '#c51f1e', '< 0.25'),
        (0.45, '#f5a361', '0.26 - 0.45'),
        (0.60, '#faf7be', '0.46 - 0.60'),
        (0.75, '#a1d193', '0.61 - 0.75'),
        ('max', '#447cb9', '>= 0.75')
    ],
    # ★★★ GNDVI를 NDVI보다 위로 이동 ★★★
    'GNDVI': [
        (0.30, '#c51f1e', '< 0.30'),
        (0.50, '#f5a361', '0.31 - 0.50'),
        (0.65, '#faf7be', '0.51 - 0.65'),
        (0.80, '#a1d193', '0.66 - 0.80'),
        ('max', '#447cb9', '>= 0.81')
    ],
    'NDVI': [
        (0.25, '#c51f1e', '< 0.25'),
        (0.45, '#f5a361', '0.26 - 0.45'),
        (0.60, '#faf7be', '0.46 - 0.60'),
        (0.75, '#a1d193', '0.61 - 0.75'),
        ('max', '#447cb9', '>= 0.76')
    ],
    'LCI': [
        (0.25, '#c51f1e', '< 0.25'),
        (0.45, '#f5a361', '0.26 - 0.45'),
        (0.65, '#faf7be', '0.46 - 0.65'),
        (0.90, '#a1d193', '0.66 - 0.90'),
        ('max', '#447cb9', '>= 0.91')
    ],
    'NDRE': [
        (0.20, '#c51f1e', '< 0.20'),
        (0.35, '#f5a361', '0.21 - 0.35'),
        (0.50, '#faf7be', '0.36 - 0.50'),
        (0.65, '#a1d193', '0.51 - 0.65'),
        ('max', '#447cb9', '>= 0.66')
    ]
}

# ------------------------- (여기부터는 수정할 필요 없습니다) -------------------------

def setup_qgis_environment():
    """QGIS 환경을 설정하는 함수"""
    py_path = os.path.join(QGIS_INSTALL_PATH, 'apps/qgis-ltr/python')
    if not os.path.isdir(py_path):
        py_path = os.path.join(QGIS_INSTALL_PATH, 'apps/qgis/python')
    sys.path.append(py_path)
    sys.path.append(os.path.join(QGIS_INSTALL_PATH, 'apps/qgis-ltr/python/plugins'))
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(QGIS_INSTALL_PATH, 'apps/Qt5/plugins')
    os.environ['QT_PLUGIN_PATH'] = os.path.join(QGIS_INSTALL_PATH, 'apps/qgis-ltr/qtplugins')
    print("QGIS 환경 설정 완료.")


def print_class_statistics(provider, rules):
    """래스터 데이터의 각 등급별 최소/최대/픽셀 수를 계산하고 출력하는 함수"""
    print("   [분석] 각 등급별 통계 계산 시작...")

    extent = provider.extent()
    width = provider.xSize()
    height = provider.ySize()

    class_pixels = {rule[2]: [] for rule in rules}
    block = provider.block(1, extent, width, height)

    # --- ★★★ 로직이 수정된 부분 ★★★ ---
    # 5개의 등급 구간 경계값을 미리 변수로 지정합니다.
    breaks = [r[0] for r in rules if isinstance(r[0], (int, float))]

    # 모든 픽셀을 하나씩 확인하며 if/elif/else로 명확하게 분류
    for i in range(width * height):
        value = block.value(i)

        if value <= breaks[0]:
            class_pixels[rules[0][2]].append(value)
        elif value <= breaks[1]:
            class_pixels[rules[1][2]].append(value)
        elif value <= breaks[2]:
            class_pixels[rules[2][2]].append(value)
        elif value <= breaks[3]:
            class_pixels[rules[3][2]].append(value)
        else:  # 나머지 모든 값 (마지막 등급)
            class_pixels[rules[4][2]].append(value)

    # 분류된 픽셀들의 통계 출력
    print("   -------------------------------------------------")
    print("   | 범례 라벨         |  픽셀 수 |   최소값   |   최대값   |")
    print("   -------------------------------------------------")
    for label, pixels in class_pixels.items():
        count = len(pixels)
        if count > 0:
            min_val = min(pixels)
            max_val = max(pixels)
            print(f"   | {label:<18}| {count:>8} | {min_val:>10.4f} | {max_val:>10.4f} |")
        else:
            # nan 대신 '-'를 출력하여 더 깔끔하게 보여줍니다.
            print(f"   | {label:<18}| {count:>8} |      -     |      -     |")
    print("   -------------------------------------------------")

def process_raster(input_path, output_path, rules):
    """단일 GeoTIFF 파일을 처리하여 PNG로 저장하는 함수"""
    from qgis.core import (QgsProject, QgsRasterLayer, QgsSingleBandPseudoColorRenderer,
                           QgsColorRampShader, QgsRasterShader, QgsMapSettings,
                           QgsMapRendererParallelJob, QgsRaster)
    from PyQt5.QtCore import QSize, QEventLoop
    from PyQt5.QtGui import QColor, QImage

    print(f"-> 처리 시작: {os.path.basename(input_path)}")
    project = QgsProject.instance()

    raster_layer = QgsRasterLayer(input_path, os.path.basename(input_path))
    if not raster_layer.isValid():
        print(f"   [오류] 레이어를 불러올 수 없습니다. 건너<binary data, 2 bytes>니다.")
        return

    project.setCrs(raster_layer.crs())
    project.addMapLayer(raster_layer)

    provider = raster_layer.dataProvider()
    # print_class_statistics(provider, rules)
    stats = provider.bandStatistics(1)
    max_value = stats.maximumValue

    # === ★★★ 변경된 부분: 새로운 규칙집으로 컬러맵 생성 ★★★ ===
    qgis_color_ramp_list = []
    for value, color, label in rules:
        if value == 'max':
            item_value = max_value
        else:
            item_value = value
        item = QgsColorRampShader.ColorRampItem(item_value, QColor(color), label)
        qgis_color_ramp_list.append(item)

    color_ramp_shader = QgsColorRampShader()
    color_ramp_shader.setColorRampType(QgsColorRampShader.Discrete)
    color_ramp_shader.setColorRampItemList(qgis_color_ramp_list)
    raster_shader = QgsRasterShader()
    raster_shader.setRasterShaderFunction(color_ramp_shader)
    renderer = QgsSingleBandPseudoColorRenderer(provider, 1, raster_shader)
    raster_layer.setRenderer(renderer)

    # 이미지 렌더링
    extent = raster_layer.extent()
    output_height = int(OUTPUT_WIDTH_PX * extent.height() / extent.width())

    settings = QgsMapSettings()
    settings.setLayers([raster_layer])
    settings.setExtent(extent)
    settings.setOutputSize(QSize(OUTPUT_WIDTH_PX, output_height))
    settings.setBackgroundColor(QColor(255, 255, 255, 0))

    job = QgsMapRendererParallelJob(settings)
    loop = QEventLoop()
    job.finished.connect(loop.quit)
    job.start()
    loop.exec_()

    image = job.renderedImage()
    image.save(output_path, "png")

    print(f"   [성공] PNG 파일 저장 완료: {os.path.basename(output_path)}")
    project.removeMapLayer(raster_layer.id())


# === ★★★ 수정된 main 함수 ★★★ ===
# === ★★★ 수정된 main 함수 ★★★ ===
def main():
    """메인 실행 함수"""
    setup_qgis_environment()
    from qgis.core import QgsApplication

    qgs = QgsApplication([], False)
    qgs.initQgis()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    search_path = os.path.join(INPUT_FOLDER, '*.tif')
    raster_files = glob.glob(search_path) + glob.glob(os.path.join(INPUT_FOLDER, '*.tiff'))

    if not raster_files:
        print(f"입력 폴더에 .tif 또는 .tiff 파일이 없습니다: {INPUT_FOLDER}")
        qgs.exitQgis()
        return

    print(f"\n총 {len(raster_files)}개의 파일을 확인합니다...")

    # 처리 대상 식생 지수 이름 목록 생성
    valid_index_names = list(CLASSIFICATION_MAP.keys())

    for file_path in raster_files:
        filename_base = os.path.basename(file_path)
        filename_upper = filename_base.upper() # 대소문자 구분 없이 비교하기 위해

        # --- 파일 이름에 유효한 식생 지수 이름이 포함되어 있는지 확인 ---
        # any() 함수를 사용하여 목록의 이름 중 하나라도 포함되면 True
        is_valid_file = any(index_name in filename_upper for index_name in valid_index_names)

        if not is_valid_file:
            print(f"-> '{filename_base}' 파일 이름에 유효한 식생 지수가 없어 건너<binary data, 2 bytes><binary data, 2 bytes><binary data, 2 bytes>니다.")
            continue # 다음 파일로 넘어감
        # --- 확인 로직 끝 ---

        # 유효한 식생 지수 파일만 아래 로직을 실행합니다.
        found_rule = False
        for index_name, rules in CLASSIFICATION_MAP.items():
            if index_name in filename_upper:
                base_name_no_ext = os.path.splitext(filename_base)[0]
                output_path = os.path.join(OUTPUT_FOLDER, f"{base_name_no_ext}.png")
                process_raster(file_path, output_path, rules)
                found_rule = True
                break # 이미 맞는 규칙을 찾았으므로 더 이상 찾지 않음

        # 이 부분은 이론상 실행되지 않아야 하지만, 안전장치로 남겨둡니다.
        if not found_rule:
            print(f"-> '{filename_base}' 파일 처리 중 예외 발생 (규칙 재탐색 실패). 건너<binary data, 2 bytes><binary data, 2 bytes><binary data, 2 bytes>니다.")

    qgs.exitQgis()
    print("\n모든 작업이 완료되었습니다.")


if __name__ == '__main__':
    main()
