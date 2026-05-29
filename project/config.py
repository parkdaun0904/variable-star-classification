# config.py

"""
Global project configuration.

This file stores settings shared across multiple modules.

Why use a separate configuration file?

- Keeps important settings in one place.
- Makes maintenance easier.
- Allows directory and model settings to be changed without editing multiple files.
- Improves project readability.

Project Structure
-----------------

Code Collection/
│
├─ project/
│  ├─ train_model.py
│  ├─ predict_new.py
│  ├─ gcvs5.txt
│  └─ asassn_catalog_full.csv
│
├─ models/
├─ results/
├─ new_data/
└─ conclusion_lightcurves/
"""

from pathlib import Path


# ------------------------------------------------------------
# Directory Paths
# ------------------------------------------------------------

# Directory containing this config.py file
PROJECT_DIR = Path(__file__).resolve().parent

# Parent directory of project/
ROOT_DIR = PROJECT_DIR.parent


# ------------------------------------------------------------
# Dataset Paths
# ------------------------------------------------------------

GCVS_PATH = PROJECT_DIR / "gcvs5.txt"
ASASSN_PATH = PROJECT_DIR / "asassn_catalog_full.csv"


# ------------------------------------------------------------
# Output Directories
# ------------------------------------------------------------

MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"
NEW_DATA_DIR = PROJECT_DIR / "new_data"
CONCLUSION_LIGHTCURVE_DIR = PROJECT_DIR / "conclusion_lightcurves"


# ------------------------------------------------------------
# Model File
# ------------------------------------------------------------

MODEL_PATH = MODEL_DIR / "gcvs_asassn_random_forest.joblib"


# ------------------------------------------------------------
# General Settings
# ------------------------------------------------------------

RANDOM_SEED = 42

# Minimum number of samples required for a label
# Labels with fewer samples are excluded from training.
MIN_SAMPLES_PER_LABEL = 30

# Number of probability rankings to display
TOP_K = 3


# ------------------------------------------------------------
# Target Variable Star Classes
# ------------------------------------------------------------

TARGET_LABELS = [
    "RR_Lyrae",
    "Cepheid",
    "Eclipsing",
    "Mira",
    "SemiRegular",
    "DeltaScuti",
]


# ------------------------------------------------------------
# Numerical Features
# ------------------------------------------------------------

NUMERIC_FEATURES = [
    "period",       # Variability period
    "mag_max",      # Brightest magnitude
    "mag_min",      # Faintest magnitude
    "mean_mag",     # Mean magnitude
    "amplitude",    # Brightness variation amplitude
    "epoch",        # Reference epoch
    "rise_time",    # Rise time ratio or eclipse duration
]


# ------------------------------------------------------------
# Categorical Features
# ------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "mag_code",
]


# ------------------------------------------------------------
# Final Feature List
# ------------------------------------------------------------

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ------------------------------------------------------------
# Delta Scuti Candidate Adjustment
# ------------------------------------------------------------

DELTA_SCUTI_PERIOD_MIN = 0.02
DELTA_SCUTI_PERIOD_MAX = 0.30

DELTA_SCUTI_AMPLITUDE_MAX = 1.00

DELTA_SCUTI_BOOST = 0.18

ECLIPSING_PENALTY_FOR_DSCT = 0.10


# ------------------------------------------------------------
# ASAS-SN Data Quality Filter
# ------------------------------------------------------------

# Minimum classification confidence accepted from ASAS-SN.
# Lower-confidence classifications are removed.
ASASSN_MIN_CLASS_PROBABILITY = 0.5