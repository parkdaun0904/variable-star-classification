# gcvs_parser.py
"""
gcvs5.txt 원본 파일을 표 형태 데이터로 바꾸는 파일

GCVS 원본 파일은 한 줄에 한 천체 정보가 들어 있고,
각 항목은 | 문자로 구분되어 있음.

이 파일에서 하는 일
1. gcvs5.txt 한 줄씩 읽기
2. 이름, 변광성 타입, 등급, 주기, 기준 시각 등 필요한 값 추출
3. '<', '>', ':', 괄호 같은 기호가 섞인 숫자값 정리
4. 밝기 변화 폭(amplitude) 계산
5. 세부 변광성 타입을 연구용 큰 라벨로 변환
"""

import re
import numpy as np
import pandas as pd

from label_mapping import map_to_group_label


def to_float(value):
    """
    GCVS 안의 숫자 문자열을 float으로 변환.
        추가 기호가 붙어있는 경우가 많기 때문.

    예:
    '< 16.'       -> 16.0
    '( 0.67 )'    -> 0.67
    '260. :'      -> 260.0
    ''            -> np.nan

        np.nan은 값이 없다(none)는 뜻.
        이를 통해 pandas와 scikit-learn에서 결측값으로 처리 가능.
    """

    if value is None:
        return np.nan

    text = str(value).strip()

    if text == "":
        return np.nan

    # 숫자 해석에 방해되는 기호 제거
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace(":", " ")
    text = text.replace("*", " ")

    # 문자열 안에서 첫 번째 숫자 부분만 찾기
    match = re.search(r"[-+]?\d+\.?\d*", text)

    if match is None:
        return np.nan

    try:
        return float(match.group(0))
    except ValueError:
        return np.nan


def is_amplitude_field(value):
    """
    최소 등급 칸이 실제 최소 등급이 아니라 진폭으로 적힌 경우 확인.

    GCVS에서는 괄호가 있는 값이 실제 최소 등급이 아니라
    밝기 변화 폭(amplitude)을 뜻하는 경우가 있음.
    """

    if value is None:
        return False

    text = str(value)

    return "(" in text and ")" in text


def parse_gcvs_txt(file_path):
    """
    gcvs5.txt 파일을 읽어서 pandas DataFrame으로 변환.
    
    DataFrame은 엑셀 표처럼 행과 열로 이루어진 자료형.
    머신러닝 학습 전에 데이터를 다루기 편하게 만들기 위해 사용.
    """

    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split("|")

            # 필드 수가 너무 적으면 정상적인 GCVS 줄이 아니라고 보고 건너뜀
            if len(parts) < 13:
                continue

            star_id = parts[0].strip()
            name = parts[1].strip()
            coord = parts[2].strip()
            var_type = parts[3].strip()

            mag_max_raw = parts[4].strip()
            mag_min_raw = parts[5].strip()
            mag_min2_raw = parts[6].strip()

            mag_code = parts[7].strip()
            epoch_raw = parts[8].strip()
            period_raw = parts[10].strip()
            rise_time_raw = parts[11].strip()
            sp_type = parts[12].strip()

            mag_max = to_float(mag_max_raw)
            mag_min = to_float(mag_min_raw)
            mag_min2 = to_float(mag_min2_raw)
            epoch = to_float(epoch_raw)
            period = to_float(period_raw)
            rise_time = to_float(rise_time_raw)

            # 등급은 숫자가 작을수록 더 밝음
            # amplitude = 최소 밝기 등급 - 최대 밝기 등급       !보통!
            if is_amplitude_field(mag_min_raw):
                amplitude = mag_min
                real_mag_min = np.nan
            else:
                real_mag_min = mag_min

                if not np.isnan(mag_max) and not np.isnan(mag_min):
                    amplitude = mag_min - mag_max
                else:
                    amplitude = np.nan

            # 첫 번째 최소 등급 칸으로 amplitude를 못 구한 경우가 생기기에
            # 두 번째 최소 등급 칸을 이용해 한 번 더 계산
            if np.isnan(amplitude):
                if is_amplitude_field(mag_min2_raw):
                    amplitude = mag_min2
                elif not np.isnan(mag_max) and not np.isnan(mag_min2):
                    amplitude = mag_min2 - mag_max

            if not np.isnan(mag_max) and not np.isnan(real_mag_min):
                mean_mag = (mag_max + real_mag_min) / 2
            else:
                mean_mag = np.nan

            # 세부 타입을 연구용 큰 그룹으로 변환
            label = map_to_group_label(var_type)

            rows.append({
                "source": "GCVS",
                "line_number": line_number,
                "star_id": star_id,
                "name": name,
                "coord": coord,
                "var_type_original": var_type,
                "label": label,
                "mag_max": mag_max,
                "mag_min": real_mag_min,
                "mag_min2": mag_min2,
                "mean_mag": mean_mag,
                "amplitude": amplitude,
                "mag_code": mag_code if mag_code else "unknown",
                "epoch": epoch,
                "period": period,
                "rise_time": rise_time,
                "sp_type": sp_type,
            })

    return pd.DataFrame(rows)