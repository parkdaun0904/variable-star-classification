# ml_pipeline.py
"""
머신러닝 학습 과정 중 사용하는 공통 함수 모음 파일

역할
1. GCVS와 ASAS-SN에서 읽은 데이터 중 실제 학습에 쓸 데이터만 고름
   - 연구 대상 6개 그룹만 사용
   - 숫자 feature가 최소 2개 이상 있는 데이터만 사용(안정적 모델 러닝 위함)
   - 한 라벨에 데이터가 너무 적으면 제외

2. 결측값 처리, 범주형 데이터 변환, Random Forest 모델을 하나의 파이프라인으로 구성
   - 비어 있는 숫자값은 중앙값으로 채움
   - mag_code 같은 문자값은 One-Hot Encoding으로 변환
   - 전처리와 모델을 묶어 학습과 예측에서 같은 처리 적용

3. 예측 결과를 확률 순위표 형태로 정리
   - top1, top2, top3 표시
   - 나머지 확률은 other로 합침
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_LABELS,
    RANDOM_SEED,
    MIN_SAMPLES_PER_LABEL,
    TOP_K,
    DELTA_SCUTI_PERIOD_MIN,
    DELTA_SCUTI_PERIOD_MAX,
    DELTA_SCUTI_AMPLITUDE_MAX,
    DELTA_SCUTI_BOOST,
    ECLIPSING_PENALTY_FOR_DSCT,
)


def make_training_data(df):
    """
    파싱된 전체 데이터에서 학습에 사용할 데이터만 선택.
    - 한 그룹에 데이터가 30개 미만이면 학습에서 제외
    - 숫자 feature 기준 1개 이하만 있으면 모델이 판단할 근거가 너무 적다고 보고 제외
    """

    data = df.copy()

    # 연구 대상 6개 그룹만 사용
    data = data[data["label"].isin(TARGET_LABELS)].copy()

    # 물리적으로 이상한 값 정리
    #   불가 : amplitude의 음수, period의 0 이하
    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = "unknown"


    # 물리적으로 이상한 값 정리
    #   불가 : amplitude의 음수, period의 0 이하 
    data.loc[data["period"] <= 0, "period"] = np.nan
    data.loc[data["amplitude"] < 0, "amplitude"] = np.nan

    # mag_code가 비어 있으면 unknown으로 통일
    data["mag_code"] = data["mag_code"].fillna("unknown")
    data["mag_code"] = data["mag_code"].replace("", "unknown")

    # 숫자 feature가 너무 부족한 데이터 제외
    numeric_count = data[NUMERIC_FEATURES].notna().sum(axis=1)
    data = data[numeric_count >= 2].copy()


    # 한 라벨의 데이터가 너무 적으면 제외
    # 최소 기준은 config.py의 MIN_SAMPLES_PER_LABEL 값 사용
    label_counts = data["label"].value_counts()
    valid_labels = label_counts[label_counts >= MIN_SAMPLES_PER_LABEL].index.tolist()

    data = data[data["label"].isin(valid_labels)].copy()

    return data


def build_model_pipeline():
    """
    전처리 과정과 Random Forest 모델을 하나로 연결.

    숫자 feature 처리
    - 비어 있는 값은 중앙값으로 채움
    - add_indicator=True 옵션으로 원래 값이 비어 있었는지도 함께 표시

    문자 feature 처리
    - mag_code처럼 문자로 된 값은 모델이 바로 계산할 수 없음
    - OneHotEncoder로 V, P, unknown 등을 숫자 벡터로 바꿈
    - handle_unknown='ignore'를 사용해 새 데이터에서 처음 보는 mag_code가 나와도 에러가 안 나게 함

    모델
    - Random Forest Classifier 사용
    - 여러 개의 결정트리를 만들고, 그 결과를 종합해서 분류
    - class_weight='balanced'로 데이터 수가 많은 라벨에만 치우치지 않게 조정
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=700,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return model


def make_top3_probability_table(model, X, names=None, true_labels=None):
    """
    Top 3 예측 확률표 생성.

    train_model.py와 predict_new.py에서 사용.
    실제 top 개수는 config.py의 TOP_K 값을 따름.
    현재 설정은 TOP_K = 3.
    """
    probabilities = model.predict_proba(X)
    classes = model.classes_

    rows = []

    for i, prob_row in enumerate(probabilities):
        order = np.argsort(prob_row)[::-1]

        if names is None:
            name = f"object_{i + 1}"
        else:
            name = names.iloc[i] if hasattr(names, "iloc") else names[i]

        row = {
            "name": name,
        }

        if true_labels is not None:
            row["true_label"] = (
                true_labels.iloc[i]
                if hasattr(true_labels, "iloc")
                else true_labels[i]
            )

        top_sum = 0.0

        for rank in range(min(TOP_K, len(classes))):
            idx = order[rank]
            label = classes[idx]
            prob = float(prob_row[idx])

            top_sum += prob

            row[f"top{rank + 1}_label"] = label
            row[f"top{rank + 1}_prob"] = round(prob, 4)
            row[f"top{rank + 1}_percent"] = f"{prob * 100:.2f}%"

        other_prob = max(0.0, 1.0 - top_sum)

        row["other_prob"] = round(other_prob, 4)
        row["other_percent"] = f"{other_prob * 100:.2f}%"

        rows.append(row)

    return pd.DataFrame(rows)


def looks_like_delta_scuti_candidate(row):
    period = pd.to_numeric(row.get("period", np.nan), errors="coerce")
    amplitude = pd.to_numeric(row.get("amplitude", np.nan), errors="coerce")

    if pd.isna(period):
        return False

    period_ok = DELTA_SCUTI_PERIOD_MIN <= period <= DELTA_SCUTI_PERIOD_MAX

    if pd.isna(amplitude):
        amplitude_ok = True
    else:
        amplitude_ok = 0 <= amplitude <= DELTA_SCUTI_AMPLITUDE_MAX

    return bool(period_ok and amplitude_ok)


def make_adjusted_probability_table(model, X, original_df, names=None):
    """
    모델 예측 확률을 보기 쉬운 표로 정리.

    출력 예
    name
    true_label
    top1_label, top1_prob, top1_percent
    top2_label, top2_prob, top2_percent
    top3_label, top3_prob, top3_percent
    other_prob, other_percent

    top1은 모델이 가장 가능성이 높다고 본 타입.
    other는 top3에 들어가지 않은 나머지 타입들의 확률 합.
    """
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    rows = []

    for i, prob_row in enumerate(probabilities):
        adjusted_prob = np.array(prob_row, dtype=float)
        original_prob = np.array(prob_row, dtype=float)

        if names is None:
            name = f"object_{i + 1}"
        else:
            name = names.iloc[i] if hasattr(names, "iloc") else names[i]

        source_row = original_df.iloc[i]
        adjustment_note = ""

        if looks_like_delta_scuti_candidate(source_row):
            if "DeltaScuti" in classes:
                dsct_idx = classes.index("DeltaScuti")
                adjusted_prob[dsct_idx] += DELTA_SCUTI_BOOST
                adjustment_note = "DeltaScuti 후보 조건(period/amplitude) 만족: 보정 적용"

            if "Eclipsing" in classes:
                ecl_idx = classes.index("Eclipsing")
                adjusted_prob[ecl_idx] = max(
                    0.0,
                    adjusted_prob[ecl_idx] - ECLIPSING_PENALTY_FOR_DSCT,
                )

            adjusted_prob = adjusted_prob / adjusted_prob.sum()

        original_order = np.argsort(original_prob)[::-1]
        adjusted_order = np.argsort(adjusted_prob)[::-1]

        row = {
            "name": name,
            "ml_top1_label": classes[original_order[0]],
            "ml_top1_prob": round(float(original_prob[original_order[0]]), 4),
            "ml_top1_percent": f"{float(original_prob[original_order[0]]) * 100:.2f}%",
            "final_top1_label": classes[adjusted_order[0]],
            "final_top1_prob": round(float(adjusted_prob[adjusted_order[0]]), 4),
            "final_top1_percent": f"{float(adjusted_prob[adjusted_order[0]]) * 100:.2f}%",
            "adjustment_note": adjustment_note,
        }

        top_sum = 0.0

        for rank in range(min(TOP_K, len(classes))):
            idx = adjusted_order[rank]
            label = classes[idx]
            prob = float(adjusted_prob[idx])

            top_sum += prob

            row[f"final_top{rank + 1}_label"] = label
            row[f"final_top{rank + 1}_prob"] = round(prob, 4)
            row[f"final_top{rank + 1}_percent"] = f"{prob * 100:.2f}%"

        other_prob = max(0.0, 1.0 - top_sum)

        row["final_other_prob"] = round(other_prob, 4)
        row["final_other_percent"] = f"{other_prob * 100:.2f}%"

        rows.append(row)

    return pd.DataFrame(rows)


def make_grouped_prediction_table(prediction_df, group_col="object_name"):
    df = prediction_df.copy()

    if group_col not in df.columns:
        return pd.DataFrame()

    if "final_top1_label" in df.columns:
        label_col = "final_top1_label"
        prob_col = "final_top1_prob"
    elif "top1_label" in df.columns:
        label_col = "top1_label"
        prob_col = "top1_prob"
    else:
        return pd.DataFrame()

    df[group_col] = df[group_col].fillna("").astype(str)
    df["_group_key"] = df[group_col].where(df[group_col].str.strip() != "", df["name"])

    rows = []

    for key, part in df.groupby("_group_key"):
        label_counts = part[label_col].value_counts()
        summary_label = label_counts.index[0]

        avg_prob = part.loc[part[label_col] == summary_label, prob_col].mean()

        rows.append(
            {
                "object_name": key,
                "n_rows": len(part),
                "summary_label": summary_label,
                "summary_label_count": int(label_counts.iloc[0]),
                "summary_avg_prob": round(float(avg_prob), 4),
                "summary_avg_percent": f"{float(avg_prob) * 100:.2f}%",
                "labels_seen": ", ".join(
                    [f"{label}:{count}" for label, count in label_counts.items()]
                ),
            }
        )

    return pd.DataFrame(rows)