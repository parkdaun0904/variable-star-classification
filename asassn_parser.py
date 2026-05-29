# asassn_parser.py
"""
ASAS-SN catalog CSV를 학습용 표 형태로 바꾸는 파일

ASAS-SN 파일에는 이미 mean_vmag, amplitude, period, variable_type 같은 값이 있음.
GCVS와 같이 학습하려면 컬럼 이름과 형태를 맞춰야 함.

이 파일에서 하는 일
1. asassn_catalog_full.csv 읽기
2. 필요한 컬럼만 가져오기
3. ASAS-SN 타입을 연구용 6개 라벨로 변환
4. GCVS 데이터와 합칠 수 있는 형태로 정리

ASAS-SN은 줄 수가 약 69만 개로 많기 때문에,
전부 손으로 한 줄씩 처리하지 않고 pandas 방식으로 한 번에 처리.
"""

import numpy as np
import pandas as pd

from config import ASASSN_MIN_CLASS_PROBABILITY
from label_mapping import map_to_group_label


def parse_asassn_csv(file_path):
    """
    ASAS-SN CSV 파일을 읽어서 학습용 DataFrame으로 변환.
    """

    use_columns = [
        "id",
        "asassn_name",
        "raj2000",
        "dej2000",
        "mean_vmag",
        "amplitude",
        "period",
        "variable_type",
        "class_probability",
        "epoch_hjd",
        "classified",
    ]

    data = pd.read_csv(
        file_path,
        usecols=lambda col: col in use_columns,
        low_memory=False,
    )

    # 숫자 컬럼 정리
    data["mean_vmag"] = pd.to_numeric(data.get("mean_vmag"), errors="coerce")
    data["amplitude"] = pd.to_numeric(data.get("amplitude"), errors="coerce")
    data["period"] = pd.to_numeric(data.get("period"), errors="coerce")
    data["epoch_hjd"] = pd.to_numeric(data.get("epoch_hjd"), errors="coerce")
    data["class_probability"] = pd.to_numeric(data.get("class_probability"), errors="coerce")

    # classified 값 정리
    # true / True / TRUE / 1 같은 값을 모두 True로 처리
    if "classified" in data.columns:
        classified_text = data["classified"].astype(str).str.lower().str.strip()
        data = data[classified_text.isin(["true", "1", "yes"])].copy()

    # 분류 확률이 너무 낮은 자료 제외
    data = data[
        data["class_probability"].isna()
        | (data["class_probability"] >= ASASSN_MIN_CLASS_PROBABILITY)
    ].copy()

    # 세부 타입을 연구용 큰 라벨로 변환
    data["label"] = data["variable_type"].apply(map_to_group_label)

    # 연구 대상 6개 그룹에 들어가지 않는 타입 제거
    data = data[data["label"].notna()].copy()

    # ASAS-SN의 mean_vmag와 amplitude를 이용해 mag_max, mag_min을 대략 계산
    # 등급은 숫자가 작을수록 밝으므로:
    # mag_max = 평균 등급 - 진폭/2
    # mag_min = 평균 등급 + 진폭/2
    data["mag_max"] = data["mean_vmag"] - data["amplitude"] / 2
    data["mag_min"] = data["mean_vmag"] + data["amplitude"] / 2

    result = pd.DataFrame({
        "source": "ASASSN",
        "line_number": np.nan,
        "star_id": data["id"],
        "name": data["asassn_name"],
        "coord": data["raj2000"].astype(str) + " " + data["dej2000"].astype(str),
        "var_type_original": data["variable_type"],
        "label": data["label"],
        "mag_max": data["mag_max"],
        "mag_min": data["mag_min"],
        "mag_min2": np.nan,
        "mean_mag": data["mean_vmag"],
        "amplitude": data["amplitude"],
        "mag_code": "V",
        "epoch": data["epoch_hjd"],
        "period": data["period"],
        "rise_time": np.nan,
        "sp_type": "",
    })

    return result