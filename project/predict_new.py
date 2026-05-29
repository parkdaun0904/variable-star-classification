# predict_new.py

"""
Predict New Variable Star Candidates

This script loads a trained machine learning model and predicts
variable star classes for new targets.

Supported Input Types
---------------------

1. Feature CSV file

Example:
    python predict_new.py --input ../new_data/sample_new_objects.csv

2. Folder containing observed light curve TXT files

Example:
    python predict_new.py --folder ../conclusion_lightcurves

3. Folder prediction with an explicitly supplied period

Example:
    python predict_new.py --folder ../conclusion_lightcurves --period 0.1234

TARGET_INFO.TXT Example
-----------------------

object_name=V799 Aur
expected_type=
period=0.0761
brightness_mode=differential_mag
filters=all
time_column=1
mag_column=2
error_column=3

Important Notes
---------------

- If period is missing, empty, "none", "nan", or invalid,
  MSE will not be calculated.

- If a valid period is available,
  MSE will be calculated for the Top 1, Top 2,
  and Top 3 predicted classes.

- Classification probability is the primary prediction result.

- MSE is only a supplementary comparison value.
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
    """
    Create a new prediction result folder.

    A timestamp is added to the folder name so that
    previous prediction results are not overwritten.
    """

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path = RESULT_DIR / f"predict_{now}"

    os.makedirs(path, exist_ok=True)

    return path


def prepare_new_data(df):
    """
    Prepare new input data for prediction.

    This function makes sure that all required feature columns exist.

    If some columns are missing, they are filled with NaN or "unknown".
    This allows the trained preprocessing pipeline to handle missing values.
    """

    data = df.copy()

    if "name" not in data.columns:
        data["name"] = [
            f"new_object_{i + 1}"
            for i in range(len(data))
        ]

    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = np.nan

    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = "unknown"

    # If amplitude is missing, try to estimate it from mag_min and mag_max.
    if "amplitude" in data.columns:
        missing_amp = data["amplitude"].isna()

        if "mag_max" in data.columns and "mag_min" in data.columns:
            data.loc[missing_amp, "amplitude"] = (
                data.loc[missing_amp, "mag_min"]
                - data.loc[missing_amp, "mag_max"]
            )

    # If mean magnitude is missing, try to estimate it from mag_max and mag_min.
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
    """
    Print warnings for light curve based prediction.

    Differential magnitude data should be interpreted carefully
    because it does not represent absolute apparent magnitude.
    """

    if "brightness_mode" in new_df.columns:
        modes = set(
            new_df["brightness_mode"]
            .dropna()
            .astype(str)
        )

        if "differential_mag" in modes:
            print("\n[Warning]")
            print("The current light curve is treated as differential_mag.")
            print(
                "The magnitude values in the TXT files are interpreted "
                "as differential or relative values, not absolute apparent magnitudes."
            )
            print(
                "Therefore, mag_max, mag_min, and mean_mag are left empty, "
                "and amplitude is used as the main brightness-related feature."
            )

    if "period" in new_df.columns:
        if new_df["period"].isna().all():
            print("\n[Warning]")
            print("No period information was found in the input data.")
            print("MSE will not be calculated.")
            print(
                "To enable MSE calculation, specify a valid period "
                "inside TARGET_INFO.TXT."
            )


def insert_metadata_columns(result_df, new_df):
    """
    Insert useful metadata columns into the prediction result table.

    These columns help connect the prediction result with
    the original target information.
    """

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
            result_df.insert(
                insert_pos,
                col,
                new_df[col].values,
            )

            insert_pos += 1

    return result_df


def add_mse_columns(result_df, new_df, n_bins=40):
    """
    Add MSE columns to the prediction result table.

    MSE is calculated for the Top 1, Top 2, and Top 3 predicted labels.

    MSE is calculated only when:
    - the original light curve file exists
    - period is valid
    - the light curve has enough data points
    """

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

        if (
            source_path is None
            or not isinstance(source_path, str)
            or not os.path.exists(source_path)
        ):
            mse_top1.append(np.nan)
            mse_top2.append(np.nan)
            mse_top3.append(np.nan)

            mse_available.append(False)
            mse_reason.append("original light curve file path not found")

            continue

        if not is_valid_period(period):
            mse_top1.append(np.nan)
            mse_top2.append(np.nan)
            mse_top3.append(np.nan)

            mse_available.append(False)
            mse_reason.append("period is missing or invalid")

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
            mse_reason.append("light curve data is empty")

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
    Print common target metadata once.
    """

    if new_df.empty:
        return

    first = new_df.iloc[0]

    print("\nTarget Information:")
    print(f"object_name     : {first.get('object_name', '')}")
    print(f"expected_type   : {first.get('expected_type', '')}")
    print(f"brightness_mode : {first.get('brightness_mode', '')}")
    print(f"period          : {first.get('period', '')}")
    print(f"period_source   : {first.get('period_source', '')}")


def print_prediction_result_compact(result_df):
    """
    Print a compact prediction result table.

    MSE columns are not shown here.

    Probability values are shown as percentages because they are
    easier to read in the terminal.
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

    show_cols = [
        col
        for col in show_cols
        if col in result_df.columns
    ]

    print("\nPrediction Results:")
    print(result_df[show_cols].to_string(index=False))


def print_mse_summary(result_df):
    """
    Print a summary of MSE calculation results.
    """

    if "mse_available" not in result_df.columns:
        return

    available_count = int(result_df["mse_available"].sum())
    total_count = len(result_df)

    print("\n" + "=" * 70)
    print("[4] MSE Summary")

    if available_count == 0:
        print("MSE available: False")

        reason = (
            result_df["mse_reason"].iloc[0]
            if total_count > 0
            else "no input data"
        )

        print(f"Reason: {reason}")

        return

    print("MSE available: True")
    print(f"Rows with MSE: {available_count}/{total_count}")

    show_cols = [
        "name",
        "top1_label",
        "top1_mse",
        "top2_label",
        "top2_mse",
        "top3_label",
        "top3_mse",
    ]

    show_cols = [
        col
        for col in show_cols
        if col in result_df.columns
    ]

    print(result_df[show_cols].to_string(index=False))


def make_average_prediction_summary(result_df):
    """
    Create an averaged summary for the same object.

    If one object has multiple light curve files, each file produces
    its own prediction.

    This function combines those predictions into a longer table and
    calculates average probability and average MSE by object and label.
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

            rows.append(
                {
                    "object_name": object_name,
                    "expected_type": expected_type,
                    "period": period,
                    "period_source": period_source,
                    "label": row.get(label_col, ""),
                    "probability": row.get(prob_col, np.nan),
                    "mse": row.get(mse_col, np.nan),
                    "rank_source": rank,
                }
            )

    long_df = pd.DataFrame(rows)

    if long_df.empty:
        return pd.DataFrame()

    summary = (
        long_df
        .groupby(
            [
                "object_name",
                "expected_type",
                "period",
                "period_source",
                "label",
            ],
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

    summary["mean_probability_percent"] = (
        summary["mean_probability"] * 100
    )

    summary = summary.sort_values(
        by=[
            "object_name",
            "mean_probability",
            "mean_mse",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    summary["final_rank"] = (
        summary
        .groupby("object_name")
        .cumcount()
        + 1
    )

    return summary


def print_average_prediction_summary(summary_df):
    """
    Print the averaged final prediction summary.
    """

    if summary_df.empty:
        return

    print("\n" + "=" * 70)
    print("[5] Averaged Prediction Summary")

    show_cols = [
        "object_name",
        "final_rank",
        "label",
        "used_count",
        "mean_probability_percent",
        "mean_mse",
        "median_mse",
    ]

    show_cols = [
        col
        for col in show_cols
        if col in summary_df.columns
    ]

    print(summary_df[show_cols].to_string(index=False))


def main():
    """
    Run prediction for either a CSV file or a light curve folder.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Predict variable star class probabilities "
            "for new candidate objects."
        )
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Path to a new feature CSV file for prediction.",
    )

    parser.add_argument(
        "--folder",
        default=None,
        help="Path to a folder containing light curve TXT files.",
    )

    parser.add_argument(
        "--period",
        default=None,
        help=(
            "Common period applied to all light curves in the folder. "
            "Example: 0.1234"
        ),
    )

    parser.add_argument(
        "--filters",
        default=None,
        help=(
            "Photometric filters to use. "
            "Examples: V or V,R or B,V,R,I"
        ),
    )

    parser.add_argument(
        "--auto-period",
        action="store_true",
        help=(
            "Estimate period using Lomb-Scargle when period is missing. "
            "This is disabled by default."
        ),
    )

    parser.add_argument(
        "--mse-bins",
        type=int,
        default=40,
        help=(
            "Number of phase bins used for MSE calculation. "
            "Default: 40"
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Run python train_model.py first."
        )

    if args.input is None and args.folder is None:
        raise ValueError(
            "No prediction input was provided.\n"
            "CSV prediction example:\n"
            "python predict_new.py --input ../new_data/file.csv\n\n"
            "Light curve folder prediction example:\n"
            "python predict_new.py --folder ../conclusion_lightcurves"
        )

    print("=" * 70)
    print("[1] Loading trained model")
    print(f"Model path: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    result_dir = make_predict_result_dir()

    print("\n" + "=" * 70)
    print("[2] Reading input data")

    if args.folder is not None:
        filters = None

        if args.filters is not None:
            filters = [
                value.strip().upper()
                for value in args.filters.split(",")
            ]

        period = safe_float(args.period)

        if np.isnan(period):
            period = None

        feature_csv_path = (
            result_dir
            / "lightcurve_features_used_for_prediction.csv"
        )

        new_df = make_feature_csv_from_folder(
            folder_path=args.folder,
            output_csv=feature_csv_path,
            period=period,
            filters=filters,
            auto_period_if_missing=args.auto_period,
        )

        print(f"Input folder: {args.folder}")
        print(
            "Selected filters: "
            f"{filters if filters is not None else 'TARGET_INFO or all'}"
        )
        print(f"Saved extracted light curve features: {feature_csv_path}")

    else:
        if not os.path.exists(args.input):
            raise FileNotFoundError(
                f"Input CSV file not found: {args.input}"
            )

        print(f"Input file: {args.input}")

        new_df = pd.read_csv(args.input)

    new_df = prepare_new_data(new_df)

    print(f"Input rows: {len(new_df):,}")

    if args.folder is not None:
        print_warning_for_lightcurve_prediction(new_df)

    X_new = new_df[ALL_FEATURES]

    print("\n" + "=" * 70)
    print("[3] Calculating Top 3 prediction probabilities")

    result_df = make_top3_probability_table(
        model=model,
        X=X_new,
        names=new_df["name"],
        true_labels=None,
    )

    result_df = insert_metadata_columns(
        result_df,
        new_df,
    )

    if args.folder is not None:
        result_df = add_mse_columns(
            result_df=result_df,
            new_df=new_df,
            n_bins=args.mse_bins,
        )

    output_path = (
        result_dir
        / "new_data_prediction_top3_with_mse.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    average_summary_df = make_average_prediction_summary(result_df)

    average_output_path = (
        result_dir
        / "average_prediction_summary.csv"
    )

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
        print(f"Saved averaged prediction summary: {average_output_path}")

    print("\n" + "=" * 70)
    print("[Completed]")
    print(f"Saved detailed prediction results: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()