# asassn_parser.py

"""
ASAS-SN Catalog Parser

This module reads the ASAS-SN variable star catalog CSV file
and converts it into the same table format used by the GCVS parser.

Why is this necessary?
----------------------

The GCVS and ASAS-SN catalogs use different column names
and slightly different data formats.

To train one machine learning model using both datasets,
their columns must be standardized.

This module:

1. Reads asassn_catalog_full.csv
2. Selects only the columns needed for this project
3. Filters out low-confidence classifications
4. Converts detailed variable types into six research labels
5. Converts ASAS-SN columns into the common training format
"""

import numpy as np
import pandas as pd

from config import ASASSN_MIN_CLASS_PROBABILITY
from label_mapping import map_to_group_label


def parse_asassn_csv(file_path):
    """
    Parse the ASAS-SN catalog CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to asassn_catalog_full.csv.

    Returns
    -------
    pandas.DataFrame
        A standardized table that can be merged with the GCVS table.

    Notes
    -----
    ASAS-SN contains many objects, so this function uses pandas
    to process the file efficiently instead of reading it line by line.
    """

    # ------------------------------------------------------------
    # Columns required for this project
    #
    # Other columns are ignored to reduce memory usage.
    # ------------------------------------------------------------

    use_columns = [
        "id",
        "asassn_name",
        "raj2000",
        "dej2000",
        "mean_vmag",
        "amplitude",
        "period",
        "variable_type",
        "class_probability",
        "epoch_hjd",
        "classified",
    ]

    data = pd.read_csv(
        file_path,
        usecols=lambda col: col in use_columns,
        low_memory=False,
    )

    # ------------------------------------------------------------
    # Convert numeric columns
    #
    # If conversion fails, the value becomes NaN.
    # This is useful because the machine learning pipeline can
    # handle missing values later.
    # ------------------------------------------------------------

    data["mean_vmag"] = pd.to_numeric(
        data.get("mean_vmag"),
        errors="coerce",
    )

    data["amplitude"] = pd.to_numeric(
        data.get("amplitude"),
        errors="coerce",
    )

    data["period"] = pd.to_numeric(
        data.get("period"),
        errors="coerce",
    )

    data["epoch_hjd"] = pd.to_numeric(
        data.get("epoch_hjd"),
        errors="coerce",
    )

    data["class_probability"] = pd.to_numeric(
        data.get("class_probability"),
        errors="coerce",
    )

    # ------------------------------------------------------------
    # Keep only rows marked as classified
    #
    # The catalog may store this value in several text formats.
    # Examples:
    # true, True, TRUE, 1, yes
    # ------------------------------------------------------------

    if "classified" in data.columns:

        classified_text = (
            data["classified"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        data = data[
            classified_text.isin(["true", "1", "yes"])
        ].copy()

    # ------------------------------------------------------------
    # Remove low-confidence classifications
    #
    # A low class_probability means the catalog itself is uncertain.
    # Using too many uncertain labels can reduce training quality.
    # ------------------------------------------------------------

    data = data[
        data["class_probability"].isna()
        | (
            data["class_probability"]
            >= ASASSN_MIN_CLASS_PROBABILITY
        )
    ].copy()

    # ------------------------------------------------------------
    # Convert detailed ASAS-SN variable types
    # into the six research labels.
    # ------------------------------------------------------------

    data["label"] = data["variable_type"].apply(
        map_to_group_label
    )

    # Remove variable types not used in this project.
    data = data[data["label"].notna()].copy()

    # ------------------------------------------------------------
    # Estimate magnitude range
    #
    # ASAS-SN provides mean_vmag and amplitude.
    #
    # Astronomical magnitude scale:
    # smaller magnitude value = brighter object
    # larger magnitude value = fainter object
    #
    # Therefore:
    #
    # mag_max = mean_vmag - amplitude / 2
    # mag_min = mean_vmag + amplitude / 2
    # ------------------------------------------------------------

    data["mag_max"] = (
        data["mean_vmag"]
        - data["amplitude"] / 2
    )

    data["mag_min"] = (
        data["mean_vmag"]
        + data["amplitude"] / 2
    )

    # ------------------------------------------------------------
    # Create standardized output table
    #
    # This format matches the output of parse_gcvs_txt().
    # ------------------------------------------------------------

    result = pd.DataFrame(
        {
            "source": "ASASSN",
            "line_number": np.nan,
            "star_id": data["id"],
            "name": data["asassn_name"],
            "coord": (
                data["raj2000"].astype(str)
                + " "
                + data["dej2000"].astype(str)
            ),
            "var_type_original": data["variable_type"],
            "label": data["label"],
            "mag_max": data["mag_max"],
            "mag_min": data["mag_min"],
            "mag_min2": np.nan,
            "mean_mag": data["mean_vmag"],
            "amplitude": data["amplitude"],
            "mag_code": "V",
            "epoch": data["epoch_hjd"],
            "period": data["period"],
            "rise_time": np.nan,
            "sp_type": "",
        }
    )

    return result