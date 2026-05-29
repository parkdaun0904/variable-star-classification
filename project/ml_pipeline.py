# ml_pipeline.py

"""
Machine Learning Pipeline Utilities

This module contains common functions used during model training
and prediction.

Main Responsibilities
---------------------

1. Select usable training data
   - Keep only the six target classes
   - Remove physically invalid values
   - Remove rows with too few numerical features
   - Remove labels with too few samples

2. Build the machine learning pipeline
   - Fill missing numerical values
   - Encode categorical values
   - Train a Random Forest classifier

3. Create prediction probability tables
   - Top 1 prediction
   - Top 2 prediction
   - Top 3 prediction
   - Remaining probability as "other"

Why use a pipeline?
-------------------

A scikit-learn Pipeline keeps preprocessing and model training
together.

This is important because the same preprocessing steps must be
applied during both training and prediction.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_LABELS,
    RANDOM_SEED,
    MIN_SAMPLES_PER_LABEL,
    TOP_K,
    DELTA_SCUTI_PERIOD_MIN,
    DELTA_SCUTI_PERIOD_MAX,
    DELTA_SCUTI_AMPLITUDE_MAX,
    DELTA_SCUTI_BOOST,
    ECLIPSING_PENALTY_FOR_DSCT,
)


def make_training_data(df):
    """
    Create the final training dataset from parsed catalog data.

    Parameters
    ----------
    df : pandas.DataFrame
        Parsed catalog data from GCVS and/or ASAS-SN.

    Returns
    -------
    pandas.DataFrame
        Cleaned training dataset.

    Filtering Rules
    ---------------

    1. Keep only target labels defined in config.py.
    2. Convert numerical features into numeric values.
    3. Remove physically impossible values.
    4. Keep only rows with at least two valid numerical features.
    5. Remove labels with fewer than MIN_SAMPLES_PER_LABEL samples.

    Why require at least two numerical features?
    --------------------------------------------

    If a row has only one or zero numerical values, the model has
    too little information to learn meaningful patterns.
    """

    data = df.copy()

    # ------------------------------------------------------------
    # Keep only the target classes used in this research.
    # ------------------------------------------------------------

    data = data[data["label"].isin(TARGET_LABELS)].copy()

    # ------------------------------------------------------------
    # Ensure all numerical feature columns exist
    # and convert them into numeric values.
    # ------------------------------------------------------------

    for col in NUMERIC_FEATURES:

        if col not in data.columns:
            data[col] = np.nan

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    # ------------------------------------------------------------
    # Ensure all categorical feature columns exist.
    # ------------------------------------------------------------

    for col in CATEGORICAL_FEATURES:

        if col not in data.columns:
            data[col] = "unknown"

    # ------------------------------------------------------------
    # Remove physically invalid values.
    #
    # Period cannot be zero or negative.
    # Amplitude cannot be negative.
    # ------------------------------------------------------------

    data.loc[data["period"] <= 0, "period"] = np.nan
    data.loc[data["amplitude"] < 0, "amplitude"] = np.nan

    # ------------------------------------------------------------
    # Standardize missing magnitude code.
    # ------------------------------------------------------------

    data["mag_code"] = data["mag_code"].fillna("unknown")
    data["mag_code"] = data["mag_code"].replace("", "unknown")

    # ------------------------------------------------------------
    # Remove rows with too few numerical features.
    # ------------------------------------------------------------

    numeric_count = data[NUMERIC_FEATURES].notna().sum(axis=1)

    data = data[numeric_count >= 2].copy()

    # ------------------------------------------------------------
    # Remove labels with too few samples.
    #
    # Very small classes can make the model unstable.
    # ------------------------------------------------------------

    label_counts = data["label"].value_counts()

    valid_labels = (
        label_counts[
            label_counts >= MIN_SAMPLES_PER_LABEL
        ]
        .index
        .tolist()
    )

    data = data[data["label"].isin(valid_labels)].copy()

    return data


def build_model_pipeline():
    """
    Build the preprocessing and Random Forest model pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete machine learning pipeline.

    Numerical Feature Processing
    ----------------------------

    Missing numerical values are replaced with the median value.

    add_indicator=True adds extra columns that indicate whether
    the original value was missing.

    This helps the model learn whether missingness itself may
    contain useful information.

    Categorical Feature Processing
    ------------------------------

    Categorical values such as mag_code cannot be directly used
    by most machine learning models.

    OneHotEncoder converts them into numerical vectors.

    handle_unknown="ignore" prevents errors when new prediction
    data contains a category not seen during training.

    Model
    -----

    RandomForestClassifier is used because it works well with
    structured tabular data and nonlinear relationships.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=700,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return model


def make_top3_probability_table(
    model,
    X,
    names=None,
    true_labels=None,
):
    """
    Create a Top 3 prediction probability table.

    This function is used by both train_model.py and predict_new.py.

    Parameters
    ----------
    model : sklearn model
        Trained machine learning model.

    X : pandas.DataFrame
        Feature table for prediction.

    names : list or pandas.Series, optional
        Object names.

    true_labels : list or pandas.Series, optional
        True labels. Used only for test data evaluation.

    Returns
    -------
    pandas.DataFrame
        Prediction probability table.

    Output Columns
    --------------

    name
    true_label
    top1_label
    top1_prob
    top1_percent
    top2_label
    top2_prob
    top2_percent
    top3_label
    top3_prob
    top3_percent
    other_prob
    other_percent
    """

    probabilities = model.predict_proba(X)
    classes = model.classes_

    rows = []

    for i, prob_row in enumerate(probabilities):

        order = np.argsort(prob_row)[::-1]

        if names is None:
            name = f"object_{i + 1}"
        else:
            name = names.iloc[i] if hasattr(names, "iloc") else names[i]

        row = {
            "name": name,
        }

        if true_labels is not None:
            row["true_label"] = (
                true_labels.iloc[i]
                if hasattr(true_labels, "iloc")
                else true_labels[i]
            )

        top_sum = 0.0

        for rank in range(min(TOP_K, len(classes))):

            idx = order[rank]
            label = classes[idx]
            prob = float(prob_row[idx])

            top_sum += prob

            row[f"top{rank + 1}_label"] = label
            row[f"top{rank + 1}_prob"] = round(prob, 4)
            row[f"top{rank + 1}_percent"] = f"{prob * 100:.2f}%"

        other_prob = max(0.0, 1.0 - top_sum)

        row["other_prob"] = round(other_prob, 4)
        row["other_percent"] = f"{other_prob * 100:.2f}%"

        rows.append(row)

    return pd.DataFrame(rows)


def looks_like_delta_scuti_candidate(row):
    """
    Check whether one object roughly matches Delta Scuti conditions.

    This is a rule-based helper function.

    It does not replace the machine learning model.
    It only checks whether the period and amplitude are located
    in a typical Delta Scuti range.

    Parameters
    ----------
    row : pandas.Series

    Returns
    -------
    bool
    """

    period = pd.to_numeric(
        row.get("period", np.nan),
        errors="coerce",
    )

    amplitude = pd.to_numeric(
        row.get("amplitude", np.nan),
        errors="coerce",
    )

    if pd.isna(period):
        return False

    period_ok = (
        DELTA_SCUTI_PERIOD_MIN
        <= period
        <= DELTA_SCUTI_PERIOD_MAX
    )

    if pd.isna(amplitude):
        amplitude_ok = True
    else:
        amplitude_ok = (
            0
            <= amplitude
            <= DELTA_SCUTI_AMPLITUDE_MAX
        )

    return bool(period_ok and amplitude_ok)


def make_adjusted_probability_table(
    model,
    X,
    original_df,
    names=None,
):
    """
    Create a probability table with optional Delta Scuti adjustment.

    Important
    ---------

    This function applies a simple rule-based correction when an
    object looks like a Delta Scuti candidate.

    This is not the main model output.

    It should be interpreted carefully as a post-processing step.

    Parameters
    ----------
    model : sklearn model
        Trained model.

    X : pandas.DataFrame
        Feature table.

    original_df : pandas.DataFrame
        Original input table before feature selection.

    names : list or pandas.Series, optional
        Object names.

    Returns
    -------
    pandas.DataFrame
        Adjusted probability table.
    """

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    rows = []

    for i, prob_row in enumerate(probabilities):

        adjusted_prob = np.array(prob_row, dtype=float)
        original_prob = np.array(prob_row, dtype=float)

        if names is None:
            name = f"object_{i + 1}"
        else:
            name = names.iloc[i] if hasattr(names, "iloc") else names[i]

        source_row = original_df.iloc[i]
        adjustment_note = ""

        if looks_like_delta_scuti_candidate(source_row):

            if "DeltaScuti" in classes:

                dsct_idx = classes.index("DeltaScuti")
                adjusted_prob[dsct_idx] += DELTA_SCUTI_BOOST

                adjustment_note = (
                    "Delta Scuti candidate condition satisfied: "
                    "period/amplitude adjustment applied"
                )

            if "Eclipsing" in classes:

                ecl_idx = classes.index("Eclipsing")

                adjusted_prob[ecl_idx] = max(
                    0.0,
                    adjusted_prob[ecl_idx]
                    - ECLIPSING_PENALTY_FOR_DSCT,
                )

            adjusted_prob = adjusted_prob / adjusted_prob.sum()

        original_order = np.argsort(original_prob)[::-1]
        adjusted_order = np.argsort(adjusted_prob)[::-1]

        row = {
            "name": name,
            "ml_top1_label": classes[original_order[0]],
            "ml_top1_prob": round(
                float(original_prob[original_order[0]]),
                4,
            ),
            "ml_top1_percent": (
                f"{float(original_prob[original_order[0]]) * 100:.2f}%"
            ),
            "final_top1_label": classes[adjusted_order[0]],
            "final_top1_prob": round(
                float(adjusted_prob[adjusted_order[0]]),
                4,
            ),
            "final_top1_percent": (
                f"{float(adjusted_prob[adjusted_order[0]]) * 100:.2f}%"
            ),
            "adjustment_note": adjustment_note,
        }

        top_sum = 0.0

        for rank in range(min(TOP_K, len(classes))):

            idx = adjusted_order[rank]
            label = classes[idx]
            prob = float(adjusted_prob[idx])

            top_sum += prob

            row[f"final_top{rank + 1}_label"] = label
            row[f"final_top{rank + 1}_prob"] = round(prob, 4)
            row[f"final_top{rank + 1}_percent"] = (
                f"{prob * 100:.2f}%"
            )

        other_prob = max(0.0, 1.0 - top_sum)

        row["final_other_prob"] = round(other_prob, 4)
        row["final_other_percent"] = f"{other_prob * 100:.2f}%"

        rows.append(row)

    return pd.DataFrame(rows)


def make_grouped_prediction_table(
    prediction_df,
    group_col="object_name",
):
    """
    Summarize prediction results by object name.

    This is useful when one astronomical target has multiple
    observation files.

    For example, one object may have several light curves from
    different filters or different observation dates.

    Parameters
    ----------
    prediction_df : pandas.DataFrame
        Prediction result table.

    group_col : str
        Column used to group results.

    Returns
    -------
    pandas.DataFrame
        Summary table by object.
    """

    df = prediction_df.copy()

    if group_col not in df.columns:
        return pd.DataFrame()

    if "final_top1_label" in df.columns:
        label_col = "final_top1_label"
        prob_col = "final_top1_prob"

    elif "top1_label" in df.columns:
        label_col = "top1_label"
        prob_col = "top1_prob"

    else:
        return pd.DataFrame()

    df[group_col] = df[group_col].fillna("").astype(str)

    df["_group_key"] = df[group_col].where(
        df[group_col].str.strip() != "",
        df["name"],
    )

    rows = []

    for key, part in df.groupby("_group_key"):

        label_counts = part[label_col].value_counts()
        summary_label = label_counts.index[0]

        avg_prob = part.loc[
            part[label_col] == summary_label,
            prob_col,
        ].mean()

        rows.append(
            {
                "object_name": key,
                "n_rows": len(part),
                "summary_label": summary_label,
                "summary_label_count": int(label_counts.iloc[0]),
                "summary_avg_prob": round(float(avg_prob), 4),
                "summary_avg_percent": f"{float(avg_prob) * 100:.2f}%",
                "labels_seen": ", ".join(
                    [
                        f"{label}:{count}"
                        for label, count in label_counts.items()
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)