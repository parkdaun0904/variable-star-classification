# lightcurve_mse.py

"""
Light Curve MSE Utilities

This module compares an observed light curve with a simple
representative template curve for each predicted variable star type.

Main Idea
---------

1. Fold the observed light curve using a known period.
2. Normalize the observed light curve.
3. Create a simple template curve for the predicted class.
4. Compare both curves using Mean Squared Error (MSE).

Important
---------

MSE is calculated only when a valid period is available.

If the period is missing, empty, "none", "nan", or invalid,
MSE is not calculated.

This template comparison is not a physical model.
It is only a supplementary shape-comparison metric.
"""

import numpy as np
import pandas as pd


def is_valid_period(period):
    """
    Check whether a period value is valid for MSE calculation.

    Invalid examples:
    - None
    - empty string
    - "none"
    - "nan"
    - "null"
    - zero or negative values
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
    Fold observation time into phase values between 0 and 1.

    Phase folding converts time data into one repeated cycle.

    Example:
    If phase = 0.25, the point is located at 25% of the cycle.
    """

    time = np.asarray(time, dtype=float)
    period = float(period)

    t0 = np.nanmin(time)

    phase = ((time - t0) / period) % 1.0

    return phase


def normalize_curve(y):
    """
    Normalize a curve using mean 0 and standard deviation 1.

    Why normalize?
    --------------

    For differential magnitude or relative brightness data,
    the exact absolute magnitude is less important than the shape
    of the variation.

    Normalization helps compare curve shapes more fairly.
    """

    y = np.asarray(y, dtype=float)

    mean = np.nanmean(y)
    std = np.nanstd(y)

    if not np.isfinite(std) or std == 0:
        return y - mean

    return (y - mean) / std


def make_phase_binned_curve(phase, mag, n_bins=40):
    """
    Divide phase into bins and calculate the mean magnitude
    inside each bin.

    This makes irregular observation points easier to compare
    with a smooth template curve.
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
    Fill empty phase bins using linear interpolation.

    If there are too few valid points, return None.
    """

    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))

    valid = np.isfinite(y)

    if np.sum(valid) < 5:
        return None

    return np.interp(x, x[valid], y[valid])


def make_template_curve(label, phase):
    """
    Create a simple representative template curve.

    Important
    ---------

    These templates are not exact physical models.

    They are simple mathematical shapes used only to compare
    whether the observed curve roughly resembles the predicted type.
    """

    label = str(label).lower()
    phase = np.asarray(phase, dtype=float)

    # Default template: sinusoidal variation
    y = np.sin(2 * np.pi * phase)

    # Delta Scuti-like template
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

    # RR Lyrae-like template
    elif "rr" in label or "lyrae" in label:
        y = (
            -np.sin(2 * np.pi * phase)
            + 0.45 * np.sin(4 * np.pi * phase)
            + 0.18 * np.sin(6 * np.pi * phase)
        )

    # Cepheid-like template
    elif "cep" in label or "cepheid" in label:
        y = (
            -np.sin(2 * np.pi * phase)
            + 0.25 * np.sin(4 * np.pi * phase)
        )

    # Eclipsing binary-like template
    elif (
        "eclipsing" in label
        or "binary" in label
        or "ea" in label
        or "eb" in label
        or "ew" in label
        or "algol" in label
        or "w uma" in label
    ):
        primary = 1.00 * np.exp(
            -((phase - 0.00) ** 2) / (2 * 0.035 ** 2)
        )

        primary += 1.00 * np.exp(
            -((phase - 1.00) ** 2) / (2 * 0.035 ** 2)
        )

        secondary = 0.55 * np.exp(
            -((phase - 0.50) ** 2) / (2 * 0.050 ** 2)
        )

        y = primary + secondary

    # Mira, semi-regular, or long-period variable template
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
    Calculate Mean Squared Error between two curves.

    Lower MSE means the two curves are more similar.
    """

    observed_y = np.asarray(observed_y, dtype=float)
    template_y = np.asarray(template_y, dtype=float)

    mask = np.isfinite(observed_y) & np.isfinite(template_y)

    if np.sum(mask) < 5:
        return np.nan

    return float(np.mean((observed_y[mask] - template_y[mask]) ** 2))


def calculate_lightcurve_mse(
    time,
    mag,
    period,
    predicted_label,
    n_bins=40,
):
    """
    Calculate MSE for one observed light curve.

    Parameters
    ----------
    time : array-like
        Observation times.

    mag : array-like
        Magnitude or differential magnitude values.

    period : float
        Known period of the target.

    predicted_label : str
        Predicted variable star type.

    n_bins : int
        Number of phase bins.

    Returns
    -------
    dict
        MSE result and status message.
    """

    if not is_valid_period(period):
        return {
            "mse_available": False,
            "mse_reason": "period is missing or invalid",
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
            "mse_reason": "not enough observation points",
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
            "mse_reason": "not enough phase-bin data",
            "mse": np.nan,
        }

    template = make_template_curve(
        predicted_label,
        phase_bin,
    )

    mse = calculate_mse(
        obs_bin,
        template,
    )

    if not np.isfinite(mse):
        return {
            "mse_available": False,
            "mse_reason": "MSE calculation failed",
            "mse": np.nan,
        }

    return {
        "mse_available": True,
        "mse_reason": "MSE calculated using the given period",
        "mse": mse,
    }


def read_observed_lightcurve(
    file_path,
    time_column=1,
    mag_column=2,
    error_column=3,
):
    """
    Read an observed light curve text file.

    Column numbers start from 1.

    Example
    -------

    time_column=1 means the first column contains time values.
    mag_column=2 means the second column contains magnitude values.

    The error column is currently read only for compatibility.
    """

    df = pd.read_csv(
        file_path,
        sep=None,
        engine="python",
    )

    time_idx = int(time_column) - 1
    mag_idx = int(mag_column) - 1

    time = pd.to_numeric(
        df.iloc[:, time_idx],
        errors="coerce",
    )

    mag = pd.to_numeric(
        df.iloc[:, mag_idx],
        errors="coerce",
    )

    valid = time.notna() & mag.notna()

    return time[valid].values, mag[valid].values