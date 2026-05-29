# config.py

"""
프로젝트 전체 설정값 모음

여러 파일에서 같이 쓰는 값을 한곳에 모아 둔 파일.
예를 들어 학습할 변광성 그룹, 사용할 feature 이름, 모델 저장 위치 등을 여기서 관리.

이렇게 따로 빼두는 이유
- 같은 값을 여러 코드에 반복해서 쓰지 않아도 됨
- 나중에 폴더명, 파일명, feature를 바꿀 때 이 파일만 수정하면 됨
- 코드 전체 구조를 한눈에 보기 쉬움

코드 파일들은 project 폴더 안에 있고,
models, results, new_data, conclusion_lightcurves 폴더는 project 바깥에 있는 구조 기준.

코드 모음/
├─ project/
│  ├─ train_model.py
│  ├─ predict_new.py
│  ├─ gcvs5.txt
│  └─ asassn_catalog_full.csv
├─ models/
├─ results/
├─ new_data/
└─ conclusion_lightcurves/
"""

from pathlib import Path

# 현재 config.py가 들어 있는 폴더
PROJECT_DIR = Path(__file__).resolve().parent
# project 폴더의 바깥 폴더
# 즉, "코드 모음" 폴더
ROOT_DIR = PROJECT_DIR.parent

GCVS_PATH = PROJECT_DIR / "gcvs5.txt"
ASASSN_PATH = PROJECT_DIR / "asassn_catalog_full.csv"

MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"
NEW_DATA_DIR = PROJECT_DIR / "new_data"
CONCLUSION_LIGHTCURVE_DIR = PROJECT_DIR / "conclusion_lightcurves"

MODEL_PATH = MODEL_DIR / "gcvs_asassn_random_forest.joblib"

RANDOM_SEED = 42
MIN_SAMPLES_PER_LABEL = 30
TOP_K = 3

# 연구에서 분류할 변광성 큰 그룹
# GCVS에는 세부 타입이 훨씬 많지만, 본 연구에서는 아래 6개 그룹만 사용
TARGET_LABELS = [
    "RR_Lyrae",
    "Cepheid",
    "Eclipsing",
    "Mira",
    "SemiRegular",
    "DeltaScuti",
]

# 숫자로 표현되는 feature
# 머신러닝 모델이 직접 계산에 사용할 수 있는 값들
NUMERIC_FEATURES = [
    "period",       # 변광 주기
    "mag_max",      # 최대 밝기 등급
    "mag_min",      # 최소 밝기 등급
    "mean_mag",     # 평균 등급
    "amplitude",    # 밝기 변화 폭
    "epoch",        # 기준 시각
    "rise_time",    # 상승 시간 비율 또는 식 지속 시간  
]

# 문자로 표현되는 feature
# 예: V, P, p, unknown 같은 광도계/필터 정보
CATEGORICAL_FEATURES = [
    "mag_code",
]

# 모델 학습과 예측에 최종적으로 사용할 전체 feature 목록
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Delta Scuti 후보 판단용 설정값
# 기본 예측에는 강제로 적용하지 않고,
# predict_new.py에서 --adjust-dsct 옵션을 켰을 때만 사용
DELTA_SCUTI_PERIOD_MIN = 0.02
DELTA_SCUTI_PERIOD_MAX = 0.30
DELTA_SCUTI_AMPLITUDE_MAX = 1.00
DELTA_SCUTI_BOOST = 0.18
ECLIPSING_PENALTY_FOR_DSCT = 0.10

# ASAS-SN 데이터에서 class_probability가 너무 낮은 자료 제외 기준
# 값이 높을수록 확실한 분류만 사용
ASASSN_MIN_CLASS_PROBABILITY = 0.5