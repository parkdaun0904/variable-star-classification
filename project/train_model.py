# train_model.py

"""
Main Training Script for the Variable Star Classifier

This script trains a machine learning model using GCVS and ASAS-SN
variable star catalog data.

Workflow
--------

1. Read the GCVS catalog file
2. Read the ASAS-SN catalog file if available
3. Merge both datasets
4. Create a cleaned training dataset
5. Split data into training and test sets
6. Train a Random Forest classifier
7. Evaluate the model using test data
8. Save the trained model and result files

This file must be executed before predict_new.py.

Run
---

python train_model.py
"""

import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from config import (
    GCVS_PATH,
    ASASSN_PATH,
    MODEL_DIR,
    RESULT_DIR,
    MODEL_PATH,
    ALL_FEATURES,
    RANDOM_SEED,
)

from gcvs_parser import parse_gcvs_txt
from asassn_parser import parse_asassn_csv

from ml_pipeline import (
    make_training_data,
    build_model_pipeline,
    make_top3_probability_table,
)


def make_run_result_dir():
    """
    Create a new result folder for each training run.

    Why?
    ----

    Each training run may produce different results.

    Saving each run in a timestamped folder prevents old results
    from being overwritten.

    Example
    -------

    results/2026-05-27_22-08-09
    """

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path = RESULT_DIR / now

    os.makedirs(path, exist_ok=True)

    return path


def print_compact_top3_examples(test_prob_df, n=10):
    """
    Print a compact Top 3 prediction example table.

    The saved CSV file contains both raw probabilities and
    percentage strings.

    For terminal output, percentage columns are easier to read.
    """

    show_cols = [
        "name",
        "true_label",
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
        if col in test_prob_df.columns
    ]

    print(f"\nExample Top 3 predictions from test data ({n} rows):")
    print(test_prob_df[show_cols].head(n).to_string(index=False))


def main():
    """
    Execute the full model training pipeline.
    """

    # ------------------------------------------------------------
    # Prepare output folders
    # ------------------------------------------------------------

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    result_dir = make_run_result_dir()

    # ------------------------------------------------------------
    # Step 1. Read catalog data
    # ------------------------------------------------------------

    print("=" * 70)
    print("[1] Reading GCVS and ASAS-SN catalog files")

    print(f"GCVS file: {GCVS_PATH}")
    df_gcvs = parse_gcvs_txt(GCVS_PATH)

    if os.path.exists(ASASSN_PATH):

        print(f"ASAS-SN file: {ASASSN_PATH}")
        df_asassn = parse_asassn_csv(ASASSN_PATH)

        df_raw = pd.concat(
            [df_gcvs, df_asassn],
            ignore_index=True,
        )

    else:

        print(f"ASAS-SN file not found: {ASASSN_PATH}")
        print("Training will continue using GCVS data only.")

        df_raw = df_gcvs

    parsed_path = result_dir / "parsed_gcvs_asassn_all.csv"

    df_raw.to_csv(
        parsed_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Total parsed rows: {len(df_raw):,}")
    print(f"Saved parsed data: {parsed_path}")

    if "source" in df_raw.columns:

        print("\nData count by source:")
        print(df_raw["source"].value_counts(dropna=False).to_string())

    # ------------------------------------------------------------
    # Step 2. Create training dataset
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("[2] Creating training dataset")

    df_ml = make_training_data(df_raw)

    training_path = result_dir / "training_dataset_grouped.csv"

    df_ml.to_csv(
        training_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Training rows used: {len(df_ml):,}")
    print(f"Saved training dataset: {training_path}")

    print("\nSample count by label:")

    label_counts = df_ml["label"].value_counts().sort_index()

    for label, count in label_counts.items():
        print(f"- {label:12s}: {count:,}")

    label_counts_path = result_dir / "label_counts.csv"

    label_counts.to_csv(
        label_counts_path,
        header=["count"],
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # Step 3. Split training and test data
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("[3] Splitting training and test data")

    X = df_ml[ALL_FEATURES]
    y = df_ml["label"]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_idx,
        test_idx,
    ) = train_test_split(
        X,
        y,
        df_ml.index,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"Training data rows: {len(X_train):,}")
    print(f"Test data rows: {len(X_test):,}")
    print("The test set is not used during training.")
    print("It is used only to evaluate model performance.")

    # ------------------------------------------------------------
    # Step 4. Train Random Forest model
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("[4] Training Random Forest model")

    model = build_model_pipeline()

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    print(f"Saved model: {MODEL_PATH}")

    # ------------------------------------------------------------
    # Step 5. Evaluate model performance
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("[5] Evaluating model with test data")

    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        digits=4,
    )

    print("\n[Classification Report]")
    print(report)

    report_path = result_dir / "classification_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    # ------------------------------------------------------------
    # Confusion Matrix
    #
    # Rows    = true labels
    # Columns = predicted labels
    # ------------------------------------------------------------

    labels_sorted = sorted(y.unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels_sorted,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels_sorted,
        columns=labels_sorted,
    )

    cm_csv_path = result_dir / "confusion_matrix.csv"

    cm_df.to_csv(
        cm_csv_path,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels_sorted,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d",
    )

    plt.title("Confusion Matrix - GCVS + ASAS-SN Classifier")
    plt.tight_layout()

    cm_png_path = result_dir / "confusion_matrix.png"

    plt.savefig(
        cm_png_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved classification report: {report_path}")
    print(f"Saved confusion matrix CSV: {cm_csv_path}")
    print(f"Saved confusion matrix image: {cm_png_path}")

    # ------------------------------------------------------------
    # Step 6. Save Top 3 prediction probabilities
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("[6] Saving Top 3 prediction probabilities for test data")

    test_objects = df_ml.loc[test_idx].reset_index(drop=True)

    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    test_prob_df = make_top3_probability_table(
        model=model,
        X=X_test_reset,
        names=test_objects["name"],
        true_labels=y_test_reset,
    )

    test_prob_path = result_dir / "test_prediction_top3.csv"

    test_prob_df.to_csv(
        test_prob_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved Top 3 test predictions: {test_prob_path}")

    print_compact_top3_examples(
        test_prob_df,
        n=10,
    )

    # ------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("[Done]")
    print(f"Result folder: {result_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()