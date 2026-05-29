"""
새로운 변광성 후보 데이터를 예측하는 실행 파일.

사용 예시:

1. CSV 파일 예측
python predict_new.py --input ../new_data/sample_new_objects.csv

2. txt 광도곡선 폴더 예측
python predict_new.py --folder ../conclusion_lightcurves

3. period를 직접 지정해서 폴더 예측 + MSE 계산
python predict_new.py --folder ../conclusion_lightcurves --period 0.1234

TARGET_INFO.TXT 예시:

object_name=V799 Aur
expected_type=
period=0.0761
brightness_mode=differential_mag
filters=all
time_column=1
mag_column=2
error_column=3

중요:
- period가 none / 빈칸 / nan / 없으면 MSE는 계산하지 않음
- period가 숫자로 있으면 top1, top2, top3 예측 타입에 대해 MSE를 계산함
"""

import argparse
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from config import (
    MODEL_PATH,
    RESULT_DIR,
    ALL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

from ml_pipeline import make_top3_probability_table

from lightcurve_features import (
    make_feature_csv_from_folder,
    read_lightcurve_txt,
    safe_float,
)

from lightcurve_mse import (
    calculate_lightcurve_mse,
    is_valid_period,
)


def make_predict_result_dir():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = RESULT_DIR / f"predict_{now}"
    os.makedirs(path, exist_ok=True)
    return path


def prepare_new_data(df):
    data = df.copy()

    if "name" not in data.columns:
        data["name"] = [f"new_object_{i + 1}" for i in range(len(data))]

    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = np.nan

    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = "unknown"

    if "amplitude" in data.columns:
        missing_amp = data["amplitude"].isna()

        if "mag_max" in data.columns and "mag_min" in data.columns:
            data.loc[missing_amp, "amplitude"] = (
                data.loc[missing_amp, "mag_min"]
                - data.loc[missing_amp, "mag_max"]
            )

    if "mean_mag" in data.columns:
        missing_mean = data["mean_mag"].isna()

        if "mag_max" in data.columns and "mag_min" in data.columns:
            data.loc[missing_mean, "mean_mag"] = (
                data.loc[missing_mean, "mag_max"]
                + data.loc[missing_mean, "mag_min"]
            ) / 2

    if "mag_code" in data.columns:
        data["mag_code"] = data["mag_code"].fillna("unknown")
        data["mag_code"] = data["mag_code"].replace("", "unknown")

    return data


def print_warning_for_lightcurve_prediction(new_df):
    if "brightness_mode" in new_df.columns:
        modes = set(new_df["brightness_mode"].dropna().astype(str))

        if "differential_mag" in modes:
            print("\n[주의]")
            print("현재 광도곡선은 differential_mag 방식으로 처리됨")
            print("txt의 밝기값을 실제 겉보기 등급이 아니라 차등등급/상대값으로 보고 있음")
            print("따라서 mag_max, mag_min, mean_mag는 비워 두고 amplitude 중심으로 사용")

    if "period" in new_df.columns:
        if new_df["period"].isna().all():
            print("\n[주의]")
            print("현재 입력 데이터에 period가 없음")
            print("따라서 MSE는 계산하지 않음")
            print("MSE가 필요하면 TARGET_INFO.TXT에 period=숫자 형태로 입력")


def insert_metadata_columns(result_df, new_df):
    insert_pos = 1

    metadata_cols = [
        "object_name",
        "expected_type",
        "brightness_mode",
        "period",
        "period_source",
        "n_points",
        "time_span",
        "phase_text",
        "memo",
    ]

    for col in metadata_cols:
        if col in new_df.columns and col not in result_df.columns:
            result_df.insert(insert_pos, col, new_df[col].values)
            insert_pos += 1

    return result_df


def add_mse_columns(result_df, new_df, n_bins=40):
    mse_top1 = []
    mse_top2 = []
    mse_top3 = []

    mse_available = []
    mse_reason = []

    for i, row in new_df.iterrows():
        source_path = row.get("source_path", None)
        period = row.get("period", np.nan)

        time_col = row.get("time_column", 1)
        mag_col = row.get("mag_column", 2)
        err_col = row.get("error_column", 3)

        if source_path is None or not isinstance(source_path, str) or not os.path.exists(source_path):
            mse_top1.append(np.nan)
            mse_top2.append(np.nan)
            mse_top3.append(np.nan)
            mse_available.append(False)
            mse_reason.append("원본 광도곡선 파일 경로 없음")
            continue

        if not is_valid_period(period):
            mse_top1.append(np.nan)
            mse_top2.append(np.nan)
            mse_top3.append(np.nan)
            mse_available.append(False)
            mse_reason.append("period 없음 또는 유효하지 않음")
            continue

        lc_df = read_lightcurve_txt(
            file_path=source_path,
            time_col=time_col,
            mag_col=mag_col,
            err_col=err_col,
        )

        if lc_df.empty:
            mse_top1.append(np.nan)
            mse_top2.append(np.nan)
            mse_top3.append(np.nan)
            mse_available.append(False)
            mse_reason.append("광도곡선 데이터 없음")
            continue

        time = lc_df["time"].values
        mag = lc_df["mag"].values

        top_mse_values = []
        local_available = False
        local_reason = ""

        for rank in [1, 2, 3]:
            label_col = f"top{rank}_label"

            if label_col not in result_df.columns:
                top_mse_values.append(np.nan)
                continue

            predicted_label = result_df.loc[i, label_col]

            mse_result = calculate_lightcurve_mse(
                time=time,
                mag=mag,
                period=period,
                predicted_label=predicted_label,
                n_bins=n_bins,
            )

            top_mse_values.append(mse_result["mse"])

            if rank == 1:
                local_available = mse_result["mse_available"]
                local_reason = mse_result["mse_reason"]

        while len(top_mse_values) < 3:
            top_mse_values.append(np.nan)

        mse_top1.append(top_mse_values[0])
        mse_top2.append(top_mse_values[1])
        mse_top3.append(top_mse_values[2])

        mse_available.append(local_available)
        mse_reason.append(local_reason)

    result_df["mse_available"] = mse_available
    result_df["mse_reason"] = mse_reason
    result_df["top1_mse"] = mse_top1
    result_df["top2_mse"] = mse_top2
    result_df["top3_mse"] = mse_top3

    return result_df


def print_prediction_metadata(new_df):
    """
    공통 메타데이터를 한 번만 출력.
    """
    if new_df.empty:
        return

    first = new_df.iloc[0]

    print("\n예측 대상 정보:")
    print(f"object_name     : {first.get('object_name', '')}")
    print(f"expected_type   : {first.get('expected_type', '')}")
    print(f"brightness_mode : {first.get('brightness_mode', '')}")
    print(f"period          : {first.get('period', '')}")
    print(f"period_source   : {first.get('period_source', '')}")


def print_prediction_result_compact(result_df):
    """
    [3] 예측 결과 출력용 간단 표.
    MSE 관련 컬럼은 여기서 출력하지 않음.
    확률도 prob 원값 대신 percent만 출력.
    """
    show_cols = [
        "name",
        "top1_label",
        "top1_percent",
        "top2_label",
        "top2_percent",
        "top3_label",
        "top3_percent",
        "other_percent",
    ]

    show_cols = [c for c in show_cols if c in result_df.columns]

    print("\n예측 결과:")
    print(result_df[show_cols].to_string(index=False))


def print_mse_summary(result_df):
    """
    MSE 계산 결과 요약 출력.
    """
    if "mse_available" not in result_df.columns:
        return

    available_count = int(result_df["mse_available"].sum())
    total_count = len(result_df)

    print("\n" + "=" * 70)
    print("[4] MSE 계산 요약")

    if available_count == 0:
        print("MSE 계산 가능 여부: False")
        print("이유:", result_df["mse_reason"].iloc[0] if total_count > 0 else "입력 없음")
        return

    print(f"MSE 계산 가능 여부: True")
    print(f"MSE 계산 가능 데이터: {available_count}/{total_count}")

    show_cols = [
        "name",
        "top1_label",
        "top1_mse",
        "top2_label",
        "top2_mse",
        "top3_label",
        "top3_mse",
    ]

    show_cols = [c for c in show_cols if c in result_df.columns]

    print(result_df[show_cols].to_string(index=False))


def make_average_prediction_summary(result_df):
    """
    같은 object_name에 대해 여러 관측 파일의 예측 확률과 MSE를 평균화한다.
    """
    rows = []

    for _, row in result_df.iterrows():
        object_name = row.get("object_name", "unknown")
        expected_type = row.get("expected_type", "")
        period = row.get("period", np.nan)
        period_source = row.get("period_source", "")

        for rank in [1, 2, 3]:
            label_col = f"top{rank}_label"
            prob_col = f"top{rank}_prob"
            mse_col = f"top{rank}_mse"

            if label_col not in result_df.columns:
                continue

            rows.append({
                "object_name": object_name,
                "expected_type": expected_type,
                "period": period,
                "period_source": period_source,
                "label": row.get(label_col, ""),
                "probability": row.get(prob_col, np.nan),
                "mse": row.get(mse_col, np.nan),
                "rank_source": rank,
            })

    long_df = pd.DataFrame(rows)

    if long_df.empty:
        return pd.DataFrame()

    summary = (
        long_df
        .groupby(
            ["object_name", "expected_type", "period", "period_source", "label"],
            dropna=False,
        )
        .agg(
            used_count=("label", "count"),
            mean_probability=("probability", "mean"),
            median_probability=("probability", "median"),
            mean_mse=("mse", "mean"),
            median_mse=("mse", "median"),
        )
        .reset_index()
    )

    summary["mean_probability_percent"] = summary["mean_probability"] * 100

    summary = summary.sort_values(
        by=["object_name", "mean_probability", "mean_mse"],
        ascending=[True, False, True],
    )

    summary["final_rank"] = (
        summary
        .groupby("object_name")
        .cumcount() + 1
    )

    return summary


def print_average_prediction_summary(summary_df):
    """
    평균화된 최종 예측 결과 출력.
    """
    if summary_df.empty:
        return

    print("\n" + "=" * 70)
    print("[5] 같은 대상 평균화 최종 결과")

    show_cols = [
        "object_name",
        "final_rank",
        "label",
        "used_count",
        "mean_probability_percent",
        "mean_mse",
        "median_mse",
    ]

    show_cols = [c for c in show_cols if c in summary_df.columns]

    print(summary_df[show_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="새로운 변광성 후보 데이터를 읽어서 타입 확률을 예측"
    )

    parser.add_argument(
        "--input",
        default=None,
        help="예측할 새 데이터 CSV 파일 경로",
    )

    parser.add_argument(
        "--folder",
        default=None,
        help="txt 광도곡선 파일들이 들어 있는 폴더 경로",
    )

    parser.add_argument(
        "--period",
        default=None,
        help="폴더 안 광도곡선에 공통으로 적용할 주기. 예: 0.1234",
    )

    parser.add_argument(
        "--filters",
        default=None,
        help="사용할 필터. 예: V 또는 V,R 또는 B,V,R,I",
    )

    parser.add_argument(
        "--auto-period",
        action="store_true",
        help="period가 없을 때 Lomb-Scargle로 자동 추정. 기본은 사용 안 함",
    )

    parser.add_argument(
        "--mse-bins",
        type=int,
        default=40,
        help="MSE 계산 시 phase bin 개수. 기본값 40",
    )

    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
            f"먼저 python train_model.py 를 실행하세요."
        )

    if args.input is None and args.folder is None:
        raise ValueError(
            "예측할 입력이 없습니다.\n"
            "CSV 예측: python predict_new.py --input ../new_data/file.csv\n"
            "txt 폴더 예측: python predict_new.py --folder ../conclusion_lightcurves"
        )

    print("=" * 70)
    print("[1] 모델 불러오기")
    print(f"모델: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    result_dir = make_predict_result_dir()

    print("\n" + "=" * 70)
    print("[2] 새 데이터 읽기")

    if args.folder is not None:
        filters = None

        if args.filters is not None:
            filters = [x.strip().upper() for x in args.filters.split(",")]

        period = safe_float(args.period)

        if np.isnan(period):
            period = None

        feature_csv_path = result_dir / "lightcurve_features_used_for_prediction.csv"

        new_df = make_feature_csv_from_folder(
            folder_path=args.folder,
            output_csv=feature_csv_path,
            period=period,
            filters=filters,
            auto_period_if_missing=args.auto_period,
        )

        print(f"입력 폴더: {args.folder}")
        print(f"사용 필터: {filters if filters is not None else 'TARGET_INFO 또는 전체'}")
        print(f"광도곡선 feature 저장: {feature_csv_path}")

    else:
        if not os.path.exists(args.input):
            raise FileNotFoundError(
                f"입력 CSV 파일을 찾을 수 없습니다: {args.input}"
            )

        print(f"입력 파일: {args.input}")
        new_df = pd.read_csv(args.input)

    new_df = prepare_new_data(new_df)

    print(f"입력 데이터 수: {len(new_df):,}")

    if args.folder is not None:
        print_warning_for_lightcurve_prediction(new_df)

    X_new = new_df[ALL_FEATURES]

    print("\n" + "=" * 70)
    print("[3] 예측 확률 Top 3 계산")

    result_df = make_top3_probability_table(
        model=model,
        X=X_new,
        names=new_df["name"],
        true_labels=None,
    )

    result_df = insert_metadata_columns(result_df, new_df)

    if args.folder is not None:
        result_df = add_mse_columns(
            result_df=result_df,
            new_df=new_df,
            n_bins=args.mse_bins,
        )

    output_path = result_dir / "new_data_prediction_top3_with_mse.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    average_summary_df = make_average_prediction_summary(result_df)

    average_output_path = result_dir / "average_prediction_summary.csv"
    average_summary_df.to_csv(
        average_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print_prediction_metadata(new_df)
    print_prediction_result_compact(result_df)

    if args.folder is not None:
        print_mse_summary(result_df)
        print_average_prediction_summary(average_summary_df)
        print(f"평균화 결과 저장: {average_output_path}")

    print("\n" + "=" * 70)
    print("[완료]")
    print(f"상세 결과 저장: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()