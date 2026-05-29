"""
관측 광도곡선과 ML 예측 타입의 대표 template 광도곡선을 MSE로 비교하는 파일.

핵심:
- TARGET_INFO.TXT 또는 feature CSV에 period가 있을 때만 MSE 계산
- period가 none / 빈칸 / NaN / 없으면 MSE 계산하지 않음
- 관측 광도곡선을 phase folding 한 뒤, 타입별 template과 비교
"""

import numpy as np
import pandas as pd


def is_valid_period(period):
    """
    period가 MSE 계산 가능한 값인지 확인.
    none, 빈칸, nan, null, 0 이하이면 False.
    """
    if period is None:
        return False

    if isinstance(period, str):
        value = period.strip().lower()
        if value in ["", "none", "nan", "null"]:
            return False

    try:
        period = float(period)
    except (ValueError, TypeError):
        return False

    return np.isfinite(period) and period > 0


def fold_phase(time, period):
    """
    관측 시간을 period 기준으로 0~1 phase로 접는다.
    """
    time = np.asarray(time, dtype=float)
    period = float(period)

    t0 = np.nanmin(time)
    phase = ((time - t0) / period) % 1.0

    return phase


def normalize_curve(y):
    """
    평균 0, 표준편차 1로 정규화.
    차등등급/상대광도 비교에서는 절대값보다 모양이 중요하므로 정규화한다.
    """
    y = np.asarray(y, dtype=float)

    mean = np.nanmean(y)
    std = np.nanstd(y)

    if not np.isfinite(std) or std == 0:
        return y - mean

    return (y - mean) / std


def make_phase_binned_curve(phase, mag, n_bins=40):
    """
    phase를 n_bins개 구간으로 나누고 각 구간의 평균 밝기값을 계산한다.
    """
    phase = np.asarray(phase, dtype=float)
    mag = np.asarray(mag, dtype=float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    binned_mag = []

    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])

        if np.sum(mask) == 0:
            binned_mag.append(np.nan)
        else:
            binned_mag.append(np.nanmean(mag[mask]))

    return bin_centers, np.asarray(binned_mag, dtype=float)


def fill_nan_by_interpolation(y):
    """
    비어 있는 phase bin은 선형보간으로 채운다.
    단, 유효한 점이 너무 적으면 None 반환.
    """
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))

    valid = np.isfinite(y)

    if np.sum(valid) < 5:
        return None

    return np.interp(x, x[valid], y[valid])


def make_template_curve(label, phase):
    """
    예측된 변광성 타입에 따른 간단한 대표 광도곡선 template 생성.

    주의:
    이건 실제 물리 모델이 아니라,
    '타입별 대표적인 변광 패턴'과 관측 곡선이 얼마나 비슷한지 보는 비교용이다.
    """
    label = str(label).lower()
    phase = np.asarray(phase, dtype=float)

    # 기본값: 사인형
    y = np.sin(2 * np.pi * phase)

    # Delta Scuti
    if (
        "delta" in label
        or "scuti" in label
        or "dsct" in label
        or "dwarf cepheid" in label
    ):
        y = (
            np.sin(2 * np.pi * phase)
            + 0.30 * np.sin(4 * np.pi * phase)
            + 0.12 * np.sin(6 * np.pi * phase)
        )

    # RR Lyrae
    elif "rr" in label or "lyrae" in label:
        y = (
            -np.sin(2 * np.pi * phase)
            + 0.45 * np.sin(4 * np.pi * phase)
            + 0.18 * np.sin(6 * np.pi * phase)
        )

    # Cepheid
    elif "cep" in label or "cepheid" in label:
        y = (
            -np.sin(2 * np.pi * phase)
            + 0.25 * np.sin(4 * np.pi * phase)
        )

    # Eclipsing Binary 전체
    elif (
        "eclipsing" in label
        or "binary" in label
        or "ea" in label
        or "eb" in label
        or "ew" in label
        or "algol" in label
        or "w uma" in label
    ):
        # phase 0과 0.5 부근에 식이 있는 모양
        primary = 1.00 * np.exp(-((phase - 0.00) ** 2) / (2 * 0.035 ** 2))
        primary += 1.00 * np.exp(-((phase - 1.00) ** 2) / (2 * 0.035 ** 2))
        secondary = 0.55 * np.exp(-((phase - 0.50) ** 2) / (2 * 0.050 ** 2))
        y = primary + secondary

    # Mira / Semiregular / Long Period
    elif (
        "mira" in label
        or "semiregular" in label
        or "semi-regular" in label
        or "lpv" in label
        or "long" in label
        or "sr" in label
    ):
        y = (
            np.sin(2 * np.pi * phase)
            + 0.35 * np.sin(4 * np.pi * phase)
        )

    return normalize_curve(y)


def calculate_mse(observed_y, template_y):
    """
    관측 곡선과 template 곡선 사이의 MSE 계산.
    """
    observed_y = np.asarray(observed_y, dtype=float)
    template_y = np.asarray(template_y, dtype=float)

    mask = np.isfinite(observed_y) & np.isfinite(template_y)

    if np.sum(mask) < 5:
        return np.nan

    return float(np.mean((observed_y[mask] - template_y[mask]) ** 2))


def calculate_lightcurve_mse(time, mag, period, predicted_label, n_bins=40):
    """
    하나의 관측 광도곡선에 대해 MSE 계산.
    """
    if not is_valid_period(period):
        return {
            "mse_available": False,
            "mse_reason": "period 없음 또는 유효하지 않음",
            "mse": np.nan,
        }

    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)

    valid = np.isfinite(time) & np.isfinite(mag)
    time = time[valid]
    mag = mag[valid]

    if len(time) < 10:
        return {
            "mse_available": False,
            "mse_reason": "관측 데이터 수 부족",
            "mse": np.nan,
        }

    phase = fold_phase(time, period)
    norm_mag = normalize_curve(mag)

    phase_bin, obs_bin = make_phase_binned_curve(
        phase=phase,
        mag=norm_mag,
        n_bins=n_bins,
    )

    obs_bin = fill_nan_by_interpolation(obs_bin)

    if obs_bin is None:
        return {
            "mse_available": False,
            "mse_reason": "phase bin 데이터 부족",
            "mse": np.nan,
        }

    template = make_template_curve(predicted_label, phase_bin)
    mse = calculate_mse(obs_bin, template)

    if not np.isfinite(mse):
        return {
            "mse_available": False,
            "mse_reason": "MSE 계산 실패",
            "mse": np.nan,
        }

    return {
        "mse_available": True,
        "mse_reason": "period 기반 MSE 계산 완료",
        "mse": mse,
    }


def read_observed_lightcurve(file_path, time_column=1, mag_column=2, error_column=3):
    """
    txt 관측 광도곡선을 읽는다.
    TARGET_INFO.TXT의 column 번호는 1부터 시작한다고 본다.
    """
    df = pd.read_csv(file_path, sep=None, engine="python")

    time_idx = int(time_column) - 1
    mag_idx = int(mag_column) - 1

    time = pd.to_numeric(df.iloc[:, time_idx], errors="coerce")
    mag = pd.to_numeric(df.iloc[:, mag_idx], errors="coerce")

    valid = time.notna() & mag.notna()

    return time[valid].values, mag[valid].values