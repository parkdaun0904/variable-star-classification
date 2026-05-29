# gcvs_parser.py

"""
GCVS Catalog Parser

This module reads the original GCVS catalog file
(gcvs5.txt) and converts it into a machine-learning-friendly
table format.

Background
----------
The original GCVS catalog is distributed as a text file.

Each row contains information about a variable star,
and fields are separated by the "|" character.

Example information stored in GCVS:

- Variable star name
- Variable type
- Magnitude range
- Period
- Epoch
- Spectral type

Machine learning models cannot directly use the raw text file.

Therefore, this module:

1. Reads the catalog line by line
2. Extracts useful fields
3. Converts text values into numerical values
4. Computes additional features
5. Maps detailed variable types into research labels
6. Returns a pandas DataFrame
"""

import re

import numpy as np
import pandas as pd

from label_mapping import map_to_group_label


def to_float(value):
    """
    Convert a GCVS numeric field into a float.

    Why is this necessary?
    ----------------------

    GCVS numeric fields often contain symbols such as:

        <
        >
        :
        *
        ( )

    Examples
    --------

    "< 16."      -> 16.0
    "(0.67)"     -> 0.67
    "260. :"     -> 260.0
    ""           -> NaN

    Parameters
    ----------
    value : str

    Returns
    -------
    float
        Numerical value or NaN.
    """

    if value is None:
        return np.nan

    text = str(value).strip()

    if text == "":
        return np.nan

    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace(":", " ")
    text = text.replace("*", " ")

    match = re.search(r"[-+]?\d+\.?\d*", text)

    if match is None:
        return np.nan

    try:
        return float(match.group(0))
    except ValueError:
        return np.nan


def is_amplitude_field(value):
    """
    Determine whether a magnitude field actually
    contains an amplitude value.

    Background
    ----------

    In some GCVS entries, the minimum magnitude
    column does not contain a true minimum magnitude.

    Instead, a value enclosed in parentheses may
    represent the variability amplitude.

    Example

        Vmax = 10.2
        (0.8)

    means:

        amplitude = 0.8

    rather than:

        minimum magnitude = 0.8

    Parameters
    ----------
    value : str

    Returns
    -------
    bool
    """

    if value is None:
        return False

    text = str(value)

    return "(" in text and ")" in text


def parse_gcvs_txt(file_path):
    """
    Parse the GCVS catalog.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    pandas.DataFrame

    Output Columns
    --------------

    source
    line_number
    star_id
    name
    coord
    var_type_original
    label
    mag_max
    mag_min
    mean_mag
    amplitude
    mag_code
    epoch
    period
    rise_time
    sp_type
    """

    rows = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for line_number, line in enumerate(file, start=1):

            line = line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split("|")

            # ----------------------------------------------------
            # Skip malformed rows
            # ----------------------------------------------------

            if len(parts) < 13:
                continue

            star_id = parts[0].strip()
            name = parts[1].strip()
            coord = parts[2].strip()
            var_type = parts[3].strip()

            mag_max_raw = parts[4].strip()
            mag_min_raw = parts[5].strip()
            mag_min2_raw = parts[6].strip()

            mag_code = parts[7].strip()
            epoch_raw = parts[8].strip()

            period_raw = parts[10].strip()
            rise_time_raw = parts[11].strip()

            sp_type = parts[12].strip()

            mag_max = to_float(mag_max_raw)
            mag_min = to_float(mag_min_raw)
            mag_min2 = to_float(mag_min2_raw)

            epoch = to_float(epoch_raw)
            period = to_float(period_raw)
            rise_time = to_float(rise_time_raw)

            # ----------------------------------------------------
            # Compute variability amplitude
            #
            # Magnitudes are inverted:
            # smaller value = brighter object
            # larger value = fainter object
            #
            # Typical formula:
            #
            # amplitude = mag_min - mag_max
            # ----------------------------------------------------

            if is_amplitude_field(mag_min_raw):

                amplitude = mag_min
                real_mag_min = np.nan

            else:

                real_mag_min = mag_min

                if (
                    not np.isnan(mag_max)
                    and not np.isnan(mag_min)
                ):
                    amplitude = mag_min - mag_max
                else:
                    amplitude = np.nan

            # ----------------------------------------------------
            # Some stars require the second minimum
            # magnitude column.
            # ----------------------------------------------------

            if np.isnan(amplitude):

                if is_amplitude_field(mag_min2_raw):
                    amplitude = mag_min2

                elif (
                    not np.isnan(mag_max)
                    and not np.isnan(mag_min2)
                ):
                    amplitude = mag_min2 - mag_max

            # ----------------------------------------------------
            # Estimate mean magnitude
            # ----------------------------------------------------

            if (
                not np.isnan(mag_max)
                and not np.isnan(real_mag_min)
            ):
                mean_mag = (
                    mag_max + real_mag_min
                ) / 2

            else:
                mean_mag = np.nan

            # ----------------------------------------------------
            # Convert detailed catalog label
            # into one of the six research classes
            # ----------------------------------------------------

            label = map_to_group_label(var_type)

            rows.append(
                {
                    "source": "GCVS",
                    "line_number": line_number,
                    "star_id": star_id,
                    "name": name,
                    "coord": coord,
                    "var_type_original": var_type,
                    "label": label,
                    "mag_max": mag_max,
                    "mag_min": real_mag_min,
                    "mag_min2": mag_min2,
                    "mean_mag": mean_mag,
                    "amplitude": amplitude,
                    "mag_code": (
                        mag_code
                        if mag_code
                        else "unknown"
                    ),
                    "epoch": epoch,
                    "period": period,
                    "rise_time": rise_time,
                    "sp_type": sp_type,
                }
            )

    return pd.DataFrame(rows)