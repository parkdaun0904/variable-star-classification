# lightcurve_features.py

"""
Light Curve Feature Extraction Utilities

This module converts observed light curve TXT files into
feature-based input data for the prediction pipeline.

Important Concept
-----------------

The TXT files inside conclusion_lightcurves/ are not training data.

They are final prediction data.

Main Responsibilities
---------------------

1. Read TARGET_INFO.TXT
2. Read observed light curve TXT files
3. Extract numerical features
4. Estimate amplitude
5. Handle differential magnitude data
6. Optionally estimate period using Lomb-Scargle
7. Save extracted features as a CSV file

TARGET_INFO.TXT Example
-----------------------

object_name=V799 Aur
expected_type=
period=
brightness_mode=differential_mag
filters=all
time_column=1
mag_column=2
error_column=3

If brightness_mode is differential_mag, the values are treated as
relative magnitude values rather than absolute apparent magnitudes.

In that case, mag_max, mag_min, and mean_mag are set to NaN,
and amplitude is used as the main brightness-related feature.
"""

import os
import glob
import re

import numpy as np
import pandas as pd

try:
    from scipy.signal import lombscargle
except ImportError:
    lombscargle = None


def safe_float(value):
    """
    Safely convert a value into float.

    Invalid values are converted to NaN.

    Invalid examples:
    - None
    - empty string
    - "none"
    - "nan"
    - "null"
    """

    try:
        if value is None:
            return np.nan

        text = str(value).strip().lower()

        if text in ["", "none", "nan", "null"]:
            return np.nan

        return float(value)

    except (ValueError, TypeError):
        return np.nan


def read_target_info(folder_path):
    """
    Read TARGET_INFO.TXT from a light curve folder.

    The file stores metadata about the observed target.

    If TARGET_INFO.TXT does not exist, default values are used.
    """

    info = {
        "object_name": "",
        "expected_type": "",
        "period": "",
        "epoch": "",
        "time_column": "1",
        "mag_column": "2",
        "error_column": "3",
        "brightness_mode": "differential_mag",
        "filters": "all",
        "memo": "",
    }

    candidates = [
        os.path.join(str(folder_path), "TARGET_INFO.TXT"),
        os.path.join(str(folder_path), "target_info.txt"),
        os.path.join(str(folder_path), "TARGET_INFO.txt"),
    ]

    info_path = None

    for path in candidates:
        if os.path.exists(path):
            info_path = path
            break

    if info_path is None:
        return info

    with open(
        info_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for line in file:
            text = line.strip()

            if not text:
                continue

            if text.startswith("#"):
                continue

            if "=" not in text:
                continue

            key, value = text.split("=", 1)

            info[key.strip()] = value.strip()

    return info


def guess_filter_from_filename(file_name):
    """
    Guess the photometric filter from a file name.

    Example
    -------

    w20250109-V799-Aur-V-LC-MAG-025.txt -> V
    """

    name = os.path.basename(file_name).upper()

    for filt in ["B", "V", "R", "I"]:
        if f"-{filt}-" in name or f"_{filt}_" in name:
            return filt

    return "unknown"


def read_lightcurve_txt(
    file_path,
    time_col=1,
    mag_col=2,
    err_col=3,
):
    """
    Read one observed light curve TXT file.

    Column numbers start from 1.

    Parameters
    ----------
    file_path : str
        Path to the light curve TXT file.

    time_col : int
        Column number for time values.

    mag_col : int
        Column number for magnitude values.

    err_col : int
        Column number for magnitude errors.

    Returns
    -------
    pandas.DataFrame
        Columns: time, mag, err
    """

    rows = []

    time_idx = int(time_col) - 1
    mag_idx = int(mag_col) - 1
    err_idx = int(err_col) - 1

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for line in file:
            text = line.strip()

            if not text:
                continue

            if text.startswith("#"):
                continue

            parts = re.split(r"\s+", text)

            if len(parts) <= max(time_idx, mag_idx):
                continue

            time = safe_float(parts[time_idx])
            mag = safe_float(parts[mag_idx])

            if np.isnan(time) or np.isnan(mag):
                continue

            if len(parts) > err_idx:
                err = safe_float(parts[err_idx])
            else:
                err = np.nan

            rows.append(
                {
                    "time": time,
                    "mag": mag,
                    "err": err,
                }
            )

    return pd.DataFrame(rows)


def calculate_phase_info(df, period, epoch):
    """
    Calculate phase coverage of the observed light curve.

    Phase is calculated only when both period and epoch are valid.

    Phase values range from 0 to 1.

    Example
    -------

    phase_text = 0.1320~0.2870

    This means that the observation covers roughly 13.2% to 28.7%
    of one variability cycle.
    """

    if (
        df.empty
        or pd.isna(period)
        or period <= 0
        or pd.isna(epoch)
    ):
        return {
            "phase_start": np.nan,
            "phase_end": np.nan,
            "phase_span": np.nan,
            "phase_text": "",
        }

    phases = ((df["time"] - epoch) / period) % 1.0

    phase_start = float(phases.min())
    phase_end = float(phases.max())
    phase_span = float(phase_end - phase_start)

    return {
        "phase_start": phase_start,
        "phase_end": phase_end,
        "phase_span": phase_span,
        "phase_text": f"{phase_start:.4f}~{phase_end:.4f}",
    }


def extract_features_from_lightcurve(
    file_path,
    period=None,
    epoch=None,
    brightness_mode="differential_mag",
    time_col=1,
    mag_col=2,
    err_col=3,
):
    """
    Extract machine learning features from one light curve file.

    If brightness_mode is apparent_mag, magnitude statistics are used
    as real apparent magnitude values.

    If brightness_mode is differential_mag, magnitude statistics are not
    treated as absolute magnitudes. In this case, mag_max, mag_min, and
    mean_mag are set to NaN.
    """

    df = read_lightcurve_txt(
        file_path=file_path,
        time_col=time_col,
        mag_col=mag_col,
        err_col=err_col,
    )

    file_name = os.path.basename(file_path)

    mag_code = guess_filter_from_filename(file_name)

    period = safe_float(period)
    epoch_from_info = safe_float(epoch)

    if df.empty:
        return {
            "name": file_name,
            "source_path": str(file_path),
            "period": period,
            "mag_max": np.nan,
            "mag_min": np.nan,
            "mean_mag": np.nan,
            "amplitude": np.nan,
            "epoch": epoch_from_info,
            "rise_time": np.nan,
            "mag_code": mag_code,
            "n_points": 0,
            "time_span": np.nan,
            "median_error": np.nan,
            "brightness_mode": brightness_mode,
            "phase_start": np.nan,
            "phase_end": np.nan,
            "phase_span": np.nan,
            "phase_text": "",
        }

    raw_min = df["mag"].min()
    raw_max = df["mag"].max()
    raw_mean = df["mag"].mean()

    amplitude = raw_max - raw_min

    best_idx = df["mag"].idxmin()
    epoch_auto = df.loc[best_idx, "time"]

    if pd.isna(epoch_from_info):
        epoch_used = epoch_auto
    else:
        epoch_used = epoch_from_info

    time_span = df["time"].max() - df["time"].min()
    median_error = df["err"].median()

    if brightness_mode == "apparent_mag":
        # In the magnitude system, smaller values mean brighter objects.
        mag_max = raw_min
        mag_min = raw_max
        mean_mag = raw_mean

    else:
        # Differential magnitude should not be interpreted
        # as absolute apparent magnitude.
        mag_max = np.nan
        mag_min = np.nan
        mean_mag = np.nan

    phase_info = calculate_phase_info(
        df,
        period=period,
        epoch=epoch_used,
    )

    return {
        "name": file_name,
        "source_path": str(file_path),
        "period": period,
        "mag_max": mag_max,
        "mag_min": mag_min,
        "mean_mag": mean_mag,
        "amplitude": amplitude,
        "epoch": epoch_used,
        "rise_time": np.nan,
        "mag_code": mag_code,
        "n_points": len(df),
        "time_span": time_span,
        "median_error": median_error,
        "brightness_mode": brightness_mode,
        "phase_start": phase_info["phase_start"],
        "phase_end": phase_info["phase_end"],
        "phase_span": phase_info["phase_span"],
        "phase_text": phase_info["phase_text"],
    }


def guess_object_name_from_files(files):
    """
    Guess the object name from light curve file names.

    This is used only when object_name is missing from TARGET_INFO.TXT.
    """

    if not files:
        return ""

    base = os.path.basename(files[0])
    name = os.path.splitext(base)[0]

    parts = re.split(r"[-_]+", name)

    parts = [
        value
        for value in parts
        if not re.match(
            r"^w?\d{8}$",
            value,
            flags=re.IGNORECASE,
        )
    ]

    stop_words = {
        "B",
        "V",
        "R",
        "I",
        "LC",
        "MAG",
    }

    object_parts = []

    for part in parts:
        upper = part.upper()

        if upper in stop_words:
            break

        if part.isdigit():
            break

        object_parts.append(part)

    return " ".join(object_parts).strip()


def estimate_period_from_lightcurve_files(
    files,
    time_col=1,
    mag_col=2,
    err_col=3,
    min_period=0.02,
    max_period=0.30,
    n_grid=30000,
):
    """
    Estimate a period using the Lomb-Scargle periodogram.

    This function is used only when:

    - period is missing
    - auto_period_if_missing is True
    - scipy is installed

    Important
    ---------

    Automatically estimated periods should be interpreted carefully.

    For final MSE comparison, it is recommended to provide a known
    period through TARGET_INFO.TXT or the --period argument.
    """

    if lombscargle is None:
        return np.nan, pd.DataFrame()

    rows = []

    for file_path in files:
        df = read_lightcurve_txt(
            file_path=file_path,
            time_col=time_col,
            mag_col=mag_col,
            err_col=err_col,
        )

        if df.empty or len(df) < 8:
            continue

        df = df.copy()

        df["mag_norm"] = df["mag"] - df["mag"].mean()

        filt = guess_filter_from_filename(file_path)

        for _, row in df.iterrows():
            rows.append(
                {
                    "time": row["time"],
                    "mag_norm": row["mag_norm"],
                    "err": row.get("err", np.nan),
                    "filter": filt,
                    "file": os.path.basename(file_path),
                }
            )

    data = pd.DataFrame(rows)

    if data.empty or len(data) < 20:
        return np.nan, pd.DataFrame()

    t = data["time"].to_numpy(dtype=float)
    y = data["mag_norm"].to_numpy(dtype=float)

    good = np.isfinite(t) & np.isfinite(y)

    t = t[good]
    y = y[good]

    if len(t) < 20:
        return np.nan, pd.DataFrame()

    t = t - t.min()
    y = y - y.mean()

    min_frequency = 1.0 / max_period
    max_frequency = 1.0 / min_period

    frequencies = np.linspace(
        min_frequency,
        max_frequency,
        n_grid,
    )

    angular_frequencies = 2.0 * np.pi * frequencies

    power = lombscargle(
        t,
        y,
        angular_frequencies,
        normalize=True,
    )

    order = np.argsort(power)[::-1]

    candidates = []
    used_periods = []

    for idx in order:
        frequency = float(frequencies[idx])
        period = float(1.0 / frequency)
        score = float(power[idx])

        # Skip periods that are nearly identical to already selected candidates.
        if any(
            abs(period - used_period) / used_period < 0.01
            for used_period in used_periods
        ):
            continue

        used_periods.append(period)

        candidates.append(
            {
                "rank": len(candidates) + 1,
                "period": period,
                "frequency_per_day": frequency,
                "power": score,
            }
        )

        if len(candidates) >= 10:
            break

    candidate_df = pd.DataFrame(candidates)

    if candidate_df.empty:
        return np.nan, candidate_df

    best_period = float(candidate_df.iloc[0]["period"])

    return best_period, candidate_df


def make_feature_csv_from_folder(
    folder_path,
    output_csv,
    period=None,
    filters=None,
    auto_period_if_missing=False,
):
    """
    Read TXT light curve files from a folder and create a feature CSV.

    Period Priority
    ---------------

    1. Period given by command-line argument
    2. Period written in TARGET_INFO.TXT
    3. Lomb-Scargle estimated period, if auto_period_if_missing=True
    4. NaN if no period is available

    MSE Recommendation
    ------------------

    MSE calculation is most reliable when period_source is either:

    - argument
    - TARGET_INFO

    Automatically estimated periods should be used carefully.
    """

    info = read_target_info(folder_path)

    info_period = safe_float(info.get("period", ""))

    period_source = ""

    if period is not None:
        period = safe_float(period)
        period_source = "argument"

    elif not np.isnan(info_period):
        period = info_period
        period_source = "TARGET_INFO"

    else:
        period = np.nan
        period_source = "missing"

    info_epoch = safe_float(info.get("epoch", ""))

    brightness_mode = (
        info
        .get("brightness_mode", "differential_mag")
        .strip()
    )

    if brightness_mode not in [
        "differential_mag",
        "apparent_mag",
    ]:
        brightness_mode = "differential_mag"

    time_col = int(info.get("time_column", 1))
    mag_col = int(info.get("mag_column", 2))
    err_col = int(info.get("error_column", 3))

    if filters is None:
        info_filters = info.get("filters", "all").strip()

        if (
            info_filters.lower() != "all"
            and info_filters != ""
        ):
            filters = [
                value.strip().upper()
                for value in info_filters.split(",")
            ]

    files = sorted(
        glob.glob(
            os.path.join(
                str(folder_path),
                "*.txt",
            )
        )
    )

    files = [
        file_path
        for file_path in files
        if os.path.basename(file_path).upper()
        not in [
            "TARGET_INFO.TXT",
            "TARGET_INFO.TXT.TXT",
        ]
    ]

    if filters is not None:
        files = [
            file_path
            for file_path in files
            if guess_filter_from_filename(file_path) in filters
        ]

    period_candidates = pd.DataFrame()

    if (
        (pd.isna(period) or period is None)
        and auto_period_if_missing
    ):
        (
            estimated_period,
            period_candidates,
        ) = estimate_period_from_lightcurve_files(
            files=files,
            time_col=time_col,
            mag_col=mag_col,
            err_col=err_col,
        )

        if not pd.isna(estimated_period):
            period = estimated_period
            period_source = "auto_lomb_scargle"

        if (
            output_csv is not None
            and not period_candidates.empty
        ):
            candidate_path = os.path.join(
                os.path.dirname(str(output_csv)),
                "period_candidates.csv",
            )

            period_candidates.to_csv(
                candidate_path,
                index=False,
                encoding="utf-8-sig",
            )

    if not info.get("object_name", "").strip():
        info["object_name"] = guess_object_name_from_files(files)

    rows = []

    for file_path in files:
        row = extract_features_from_lightcurve(
            file_path=file_path,
            period=period,
            epoch=info_epoch,
            brightness_mode=brightness_mode,
            time_col=time_col,
            mag_col=mag_col,
            err_col=err_col,
        )

        row["object_name"] = info.get("object_name", "")
        row["expected_type"] = info.get("expected_type", "")
        row["memo"] = info.get("memo", "")
        row["period_source"] = period_source

        row["time_column"] = time_col
        row["mag_column"] = mag_col
        row["error_column"] = err_col

        rows.append(row)

    result = pd.DataFrame(rows)

    if output_csv is not None:
        result.to_csv(
            output_csv,
            index=False,
            encoding="utf-8-sig",
        )

    return result