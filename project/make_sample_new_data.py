# make_sample_new_data.py

"""
Create Sample Prediction Data

This utility script generates an example CSV file that can be
used to test the prediction pipeline.

Purpose
-------

A trained model requires feature-based input data.

New users may not yet have real observations or extracted
features available.

This script creates a small example dataset so that the
prediction workflow can be tested immediately.

Example Workflow
----------------

1. Train model

    python train_model.py

2. Create sample prediction file

    python make_sample_new_data.py

3. Run prediction

    python predict_new.py --input ../new_data/sample_new_objects.csv
"""

from pathlib import Path

import pandas as pd

from config import NEW_DATA_DIR


def create_sample_dataframe():
    """
    Create a small example feature table.

    Returns
    -------
    pandas.DataFrame
        Example objects with realistic feature values.
    """

    rows = [
        {
            "name": "sample_rr_lyrae",
            "period": 0.56,
            "mag_max": 11.8,
            "mag_min": 12.6,
            "mean_mag": 12.2,
            "amplitude": 0.8,
            "epoch": 2459000.0,
            "rise_time": 0.15,
            "mag_code": "V",
        },
        {
            "name": "sample_cepheid",
            "period": 5.24,
            "mag_max": 9.8,
            "mag_min": 10.6,
            "mean_mag": 10.2,
            "amplitude": 0.8,
            "epoch": 2459000.0,
            "rise_time": 0.30,
            "mag_code": "V",
        },
        {
            "name": "sample_dsct",
            "period": 0.12,
            "mag_max": 12.4,
            "mag_min": 12.8,
            "mean_mag": 12.6,
            "amplitude": 0.4,
            "epoch": 2459000.0,
            "rise_time": 0.20,
            "mag_code": "V",
        },
    ]

    return pd.DataFrame(rows)


def main():
    """
    Generate the example CSV file.
    """

    NEW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        NEW_DATA_DIR
        / "sample_new_objects.csv"
    )

    df = create_sample_dataframe()

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 60)
    print("Sample prediction dataset created.")
    print(f"Output file: {output_path}")
    print(f"Number of rows: {len(df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()