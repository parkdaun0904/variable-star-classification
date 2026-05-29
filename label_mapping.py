# label_mapping.py
"""
GCVS와 ASAS-SN의 세부 변광성 타입을 연구용 큰 그룹으로 묶는 파일

원본 카탈로그에는 RRAB, RRC, EA, EB, SR, SRA, DSCT처럼 타입이 잘게 나뉨.
그대로 학습하면 라벨이 너무 많아져서 모델이 헷갈릴 수 있음.
또한 각 라벨별 데이터 수가 부족해질 수 있음.

그래서 이번 연구에서는 아래 6개 그룹으로 묶어서 사용.
- RR_Lyrae
- Cepheid
- Eclipsing
- Mira
- SemiRegular
- DeltaScuti
"""


def normalize_var_type(raw_type):
    """
    변광성 타입 문자열을 비교하기 쉬운 형태로 정리.

    예:
    'RRAB'   -> 'RRAB'
    'M:'     -> 'M'
    'EA/RS'  -> 'EA/RS'
    ' dsct ' -> 'DSCT'
    """

    if raw_type is None:
        return ""

    text = str(raw_type).strip().upper()
    text = text.replace(":", "")
    text = text.replace(" ", "")

    return text


def map_to_group_label(raw_type):
    """
    세부 변광성 타입을 연구용 큰 라벨로 변환.

    변환 결과:
    - RR_Lyrae
    - Cepheid
    - Eclipsing
    - Mira
    - SemiRegular
    - DeltaScuti

    연구 대상에 포함되지 않는 타입은 None으로 처리.
    None이 된 데이터는 학습 단계에서 제외.
    """

    t = normalize_var_type(raw_type)

    if not t:
        return None

    # 복합 타입이면 앞쪽 타입을 우선 사용
    # 예: EA/RS -> EA
    primary = t.split("/")[0]
    primary = primary.split("+")[0]

    # RR Lyrae 계열
    if primary.startswith("RR"):
        return "RR_Lyrae"

    # Cepheid 계열
    if (
        primary.startswith("DCEP")
        or primary.startswith("CEP")
        or primary.startswith("CW")
        or primary.startswith("T2CEP")
    ):
        return "Cepheid"

    # 식쌍성 계열
    if (
        primary == "E"
        or primary.startswith("EA")
        or primary.startswith("EB")
        or primary.startswith("EW")
        or primary.startswith("ELL")
    ):
        return "Eclipsing"

    # Mira형
    if primary == "M":
        return "Mira"

    # 준규칙 변광성
    if primary.startswith("SR"):
        return "SemiRegular"

    # Delta Scuti 계열
    if primary.startswith("DSCT") or primary.startswith("HADS"):
        return "DeltaScuti"

    # 여기에 포함되지 않는 타입은 None으로 처리
    return None