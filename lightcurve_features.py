"""
결론용 txt 광도곡선 파일을 예측용 feature로 바꾸는 파일.

핵심:
- conclusion_lightcurves 안의 txt는 학습용이 아니라 최종 확인용 예측 데이터
- TARGET_INFO.TXT에서 period를 읽음
- period가 none / 빈칸 / NaN이면 period는 NaN으로 둠
- brightness_mode=differential_mag이면 평균등급/최대등급/최소등급은 NaN 처리
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
    값을 float으로 변환.
    none, 빈칸, nan, null이면 NaN 반환.
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
    conclusion_lightcurves 폴더 안의 TARGET_INFO.TXT 읽기.
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

    with open(info_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            text = line.strip()

            if not text or text.startswith("#") or "=" not in text:
                continue

            key, value = text.split("=", 1)
            info[key.strip()] = value.strip()

    return info


def guess_filter_from_filename(file_name):
    """
    파일 이름에서 필터 추정.
    예:
    w20250109-V799-Aur-V-LC-MAG-025.txt -> V
    """
    name = os.path.basename(file_name).upper()

    for filt in ["B", "V", "R", "I"]:
        if f"-{filt}-" in name or f"_{filt}_" in name:
            return filt

    return "unknown"


def read_lightcurve_txt(file_path, time_col=1, mag_col=2, err_col=3):
    """
    txt 광도곡선 파일 읽기.

    time_col, mag_col, err_col은 1부터 시작하는 번호.
    """
    rows = []

    time_idx = int(time_col) - 1
    mag_idx = int(mag_col) - 1
    err_idx = int(err_col) - 1

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            text = line.strip()

            if not text or text.startswith("#"):
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

            rows.append({
                "time": time,
                "mag": mag,
                "err": err,
            })

    return pd.DataFrame(rows)


def calculate_phase_info(df, period, epoch):
    """
    period와 epoch가 있을 때 관측 phase 범위 계산.
    """
    if df.empty or pd.isna(period) or period <= 0 or pd.isna(epoch):
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
    광도곡선 하나에서 feature 추출.
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
        # 등급은 숫자가 작을수록 밝음
        mag_max = raw_min
        mag_min = raw_max
        mean_mag = raw_mean
    else:
        # 차등등급은 절대 등급처럼 해석하지 않음
        mag_max = np.nan
        mag_min = np.nan
        mean_mag = np.nan

    phase_info = calculate_phase_info(df, period=period, epoch=epoch_used)

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
    파일 이름에서 object name 대략 추정.
    """
    if not files:
        return ""

    base = os.path.basename(files[0])
    name = os.path.splitext(base)[0]
    parts = re.split(r"[-_]+", name)

    parts = [
        x for x in parts
        if not re.match(r"^w?\d{8}$", x, flags=re.IGNORECASE)
    ]

    stop_words = {"B", "V", "R", "I", "LC", "MAG"}
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
    period가 없을 때 Lomb-Scargle로 대략 추정.
    단, MSE는 TARGET_INFO 또는 argument로 period가 명시된 경우에만 쓰는 것을 권장.
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
            rows.append({
                "time": row["time"],
                "mag_norm": row["mag_norm"],
                "err": row.get("err", np.nan),
                "filter": filt,
                "file": os.path.basename(file_path),
            })

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

    frequencies = np.linspace(min_frequency, max_frequency, n_grid)
    angular_frequencies = 2.0 * np.pi * frequencies

    power = lombscargle(t, y, angular_frequencies, normalize=True)
    order = np.argsort(power)[::-1]

    candidates = []
    used_periods = []

    for idx in order:
        frequency = float(frequencies[idx])
        period = float(1.0 / frequency)
        score = float(power[idx])

        if any(abs(period - p) / p < 0.01 for p in used_periods):
            continue

        used_periods.append(period)

        candidates.append({
            "rank": len(candidates) + 1,
            "period": period,
            "frequency_per_day": frequency,
            "power": score,
        })

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
    폴더 안의 txt 파일들을 읽어서 feature CSV 생성.

    우선순위:
    1. 명령어 --period
    2. TARGET_INFO.TXT의 period
    3. auto_period_if_missing=True일 때 Lomb-Scargle 추정값
    4. 없으면 NaN

    MSE는 period_source가 argument 또는 TARGET_INFO일 때만 권장.
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

    brightness_mode = info.get("brightness_mode", "differential_mag").strip()

    if brightness_mode not in ["differential_mag", "apparent_mag"]:
        brightness_mode = "differential_mag"

    time_col = int(info.get("time_column", 1))
    mag_col = int(info.get("mag_column", 2))
    err_col = int(info.get("error_column", 3))

    if filters is None:
        info_filters = info.get("filters", "all").strip()

        if info_filters.lower() != "all" and info_filters != "":
            filters = [x.strip().upper() for x in info_filters.split(",")]

    files = sorted(glob.glob(os.path.join(str(folder_path), "*.txt")))

    files = [
        f for f in files
        if os.path.basename(f).upper() not in ["TARGET_INFO.TXT", "TARGET_INFO.TXT.TXT"]
    ]

    if filters is not None:
        files = [
            f for f in files
            if guess_filter_from_filename(f) in filters
        ]

    period_candidates = pd.DataFrame()

    if (pd.isna(period) or period is None) and auto_period_if_missing:
        estimated_period, period_candidates = estimate_period_from_lightcurve_files(
            files=files,
            time_col=time_col,
            mag_col=mag_col,
            err_col=err_col,
        )

        if not pd.isna(estimated_period):
            period = estimated_period
            period_source = "auto_lomb_scargle"

        if output_csv is not None and not period_candidates.empty:
            candidate_path = os.path.join(
                os.path.dirname(str(output_csv)),
                "period_candidates.csv",
            )
            period_candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")

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
        result.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return result