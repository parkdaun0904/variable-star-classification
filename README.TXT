# Variable Star Machine Learning Classification Project

========================================================================
1. Project Overview
This project classifies variable stars using a machine learning model trained on catalog-based variable star data.
The main research topic is:
    "Performance Comparison between Traditional Period Analysis and Machine Learning-Based Variable Star Classification under Irregular Light Curve Data"

The machine learning part of the research is implemented in Python.
The model does not directly classify light curve images. Instead, it uses numerical and categorical features extracted from variable star catalogs or observed light curve files.
Main features used by the model:
    * period
    * mag_max
    * mag_min
    * mean_mag
    * amplitude
    * epoch
    * rise_time
    * mag_code
The final classifier is based on a Random Forest model.
The trained model predicts the probability that a new target belongs to each variable star class.


========================================================================
2. Project Folder Structure
The project was developed using the following directory structure.
Code Collection/
│
├─ project/
│  ├─ train_model.py
│  ├─ predict_new.py
│  ├─ gcvs_parser.py
│  ├─ asassn_parser.py
│  ├─ lightcurve_features.py
│  ├─ lightcurve_mse.py
│  ├─ ml_pipeline.py
│  ├─ label_mapping.py
│  ├─ config.py
│  ├─ make_sample_new_data.py
│  ├─ requirements.txt
│  ├─ README.TXT
│  ├─ gcvs5.txt
│  └─ asassn_catalog_full.csv
│
├─ conclusion_lightcurves/
│  ├─ TARGET_INFO.TXT
│  └─ *.txt
│
├─ models/
├─ results/
└─ new_data/

Folder Description
    project/
    Contains all source code files and training datasets.
    This is the main working directory of the project.
    All Python scripts should be executed from inside this folder.

    conclusion_lightcurves/
    Contains observed light curve files used for final prediction.
    These files are not used for training.
    They are used only for prediction and evaluation.
    
    models/
    Stores trained machine learning models.
    Example:
        models/gcvs_asassn_random_forest.joblib
    
    results/
    Stores all training and prediction outputs.
    A new timestamped folder is automatically created whenever training or prediction is executed.
    
    new_data/
    Stores CSV files used for prediction.
    Example:    
        new_data/sample_new_objects.csv

Important
The following files must exist inside the project folder before training:    
    * gcvs5.txt
    * asassn_catalog_full.csv
Expected structure:
        
    Code Collection/
    │
    ├─ project/
    │  ├─ train_model.py
    │  ├─ gcvs5.txt
    │  └─ asassn_catalog_full.csv
    Do not place these datasets inside models/, results/, or other folders.
    The training scripts expect them to be located directly inside project/.


========================================================================
3. Required Datasets
    [1] GCVS
    General Catalogue of Variable Stars.
    A traditional variable star catalog that contains information such as:
        * variable star name
        * variable type
        * period
        * magnitude
        * amplitude-related information
    Dataset page:
        http://www.sai.msu.su/gcvs/gcvs/gcvs5/htm/
    You must download this dataset yourself.
    Expected file name:
        gcvs5.txt
    The file must be placed directly inside the project folder.

[2] ASAS-SN
    All-Sky Automated Survey for Supernovae.
    A large catalog containing hundreds of thousands of variable stars.
    Dataset page:          
        https://asas-sn.osu.edu/variables
    You must download this dataset yourself.
    Expected file name: 
        asassn_catalog_full.csv
    The file must be placed directly inside the project folder.
    To improve training quality, low-confidence classifications are removed.

Current setting:
    ASASSN_MIN_CLASS_PROBABILITY = 0.5
Only reasonably reliable classifications are used for training.


========================================================================
4. Target Variable Star Classes
This project groups detailed catalog labels into six larger research classes.
The six target classes are:
    * RR_Lyrae
    * Cepheid
    * Eclipsing
    * Mira
    * SemiRegular
    * DeltaScuti
Although GCVS and ASAS-SN contain many more detailed variable star types, using too many classes often reduces model stability and increases class imbalance.
Therefore, detailed labels are simplified into these six major groups.


========================================================================
5. Main File Descriptions
config.py
Stores global project settings.
Examples:
    * dataset locations
    * model locations
    * result locations
    * feature lists
    * Random Forest settings
    * classification thresholds

label_mapping.py
Converts detailed catalog labels into the six research classes.
Examples:
        
    RRAB -> RR_Lyrae
    EA -> Eclipsing
    DSCT -> DeltaScuti

gcvs_parser.py
    Reads gcvs5.txt and converts it into a machine learning table format.
    Extracted information includes:
        * variable type
        * period
        * magnitude
        * amplitude
        * epoch
        * magnitude code

asassn_parser.py
    Reads ASAS-SN data and converts it into the same format used by GCVS.
    This allows both datasets to be combined into a single training dataset.
    ml_pipeline.py
    Contains the machine learning pipeline.
    Functions include:
        * data filtering
        * missing value handling
        * categorical encoding
        * Random Forest construction
        * probability table generation
train_model.py
    Main training script.
    Workflow:
        1. Read GCVS data
        2. Read ASAS-SN data
        3. Merge datasets
        4. Create training dataset
        5. Split training and test sets
        6. Train the Random Forest model
        7. Evaluate performance
        8. Save results and model files

predict_new.py
    Predicts variable star classes for new targets.
    Supports:   
        * CSV feature files
        * light curve TXT files
    Outputs Top 3 predicted classes and probabilities.

lightcurve_features.py
    Converts observed light curve TXT files into machine learning features.
    Reads TARGET_INFO.TXT and extracts:        
        * period
        * amplitude
        * filter information
        * phase information
        * observation statistics

lightcurve_mse.py
    Calculates Mean Squared Error (MSE) between observed light curves and template light curves.
    MSE is calculated only when a valid period exists.

make_sample_new_data.py
    Creates an example CSV file for testing the prediction pipeline.
    requirements.txt
    Lists required Python packages.


========================================================================
6. Installation
Install all required packages:
    
pip install -r requirements.txt
Required packages:
    * numpy
    * pandas
    * scikit-learn
    * scipy
    * matplotlib
    * joblib


========================================================================
7. Model Training
Run:    
    python train_model.py
Before training, verify that the following files exist:
    * gcvs5.txt
    * asassn_catalog_full.csv
ASAS-SN is optional.
If ASAS-SN is unavailable, training can still proceed using only GCVS data.
However, using both datasets is strongly recommended because it significantly increases the amount of training data.

Training outputs:
models/gcvs_asassn_random_forest.joblib
results/YYYY-MM-DD_HH-MM-SS/
├─ parsed_gcvs_asassn_all.csv
├─ training_dataset_grouped.csv
├─ label_counts.csv
├─ classification_report.txt
├─ confusion_matrix.csv
├─ confusion_matrix.png
└─ test_prediction_top3.csv


========================================================================
8. Why Training and Test Sets Are Separated
The dataset is divided into:
    * 80% training data
    * 20% test data
Training data teaches the model.
Test data evaluates how well the model performs on data that it has never seen before.
This provides a more objective measurement of model performance.


========================================================================
9. Why Random Forest Was Used
Random Forest was selected because it performs well on structured astronomical catalog data.
Advantages include:
    * Works well with tabular feature data
    * Handles nonlinear relationships between variables
    * Robust against missing values
    * Stable performance on large datasets
    * Produces probability-based classifications
    * Easy to interpret compared to deep learning models
This project focuses on feature-based classification rather than image-based classification.


========================================================================
10. Prediction Using CSV Files
Example:
    python predict_new.py --input ../new_data/sample_new_objects.csv
The CSV file should contain machine learning features such as:    
    * period
    * amplitude
    * mag_max
    * mag_min
    * mean_mag
The model will output Top 3 predicted classes and probabilities.


========================================================================
11. Prediction Using Light Curve TXT Files
Example:
    python predict_new.py --folder ../conclusion_lightcurves
The folder must contain:
            
    * TARGET_INFO.TXT
    * one or more TXT light curve files
The prediction pipeline automatically extracts features from the light curves and performs classification.


========================================================================
12. TARGET_INFO.TXT Format
Every prediction folder should contain a TARGET_INFO.TXT file.
    Example:    
    object_name=V799 Aur
    expected_type=
    period=
    brightness_mode=differential_mag
    filters=all
    time_column=1
    mag_column=2
    error_column=3
Parameter Description
object_name
Target object name.
expected_type
Optional reference label.
period
Known period of the target.
If empty, period-dependent calculations may be skipped.
brightness_mode
    Supported values:
    * differential_mag
    * apparent_mag
Most student observations use differential_mag.
filters
Photometric filters to use.
Examples:
    V
    B,V,R
    all
    time_column
    Column number containing observation time.
    mag_column
    Column number containing magnitude values.
    error_column
    Column number containing magnitude uncertainty values.


========================================================================
13. MSE Calculation
The project can compare observed light curves with representative template light curves using Mean Squared Error (MSE).
    Requirements:
    * valid period
    * sufficient observations
    * successful phase folding
If period is missing, invalid, or empty, MSE is not calculated.
MSE is used only as a supplementary comparison metric.
Classification probability remains the primary result.


========================================================================
14. Research Background
This project was developed for the 2026 College of Natural Sciences Research Festival.
The goal of the research is to compare:
    1. Traditional Lomb-Scargle Period Analysis (LSP)
    2. Machine Learning-Based Variable Star Classification
under irregular observational conditions.
The research investigates whether machine learning can maintain stable classification performance when observation quality becomes poor or data are incomplete.
Training data were collected from:
    * GCVS
    * ASAS-SN
After preprocessing and filtering, approximately 450,000 variable star samples were used for training.


========================================================================
15. Current Limitations
This project does not directly classify light curve images.
Current workflow:
    Light Curve
    -> Feature Extraction
    -> Machine Learning Model
    -> Variable Star Classification
The model relies on extracted features rather than complete light curve morphology.
Performance may decrease when:        
    * period information is unavailable
    * observation coverage is limited
    * multiple variable star types share similar feature values


========================================================================
16. Future Work
Possible future improvements include:
    * Additional Cepheid and Delta Scuti training data
    * More variable star classes
    * CNN-based light curve image classification
    * Deep learning time-series models
    * Improved automatic period estimation
    * Multi-filter light curve analysis


========================================================================
17. Citation
If you use this project in research, presentations, or educational work, please cite the original project and the source catalogs:
    * GCVS
    * ASAS-SN
Please also acknowledge the 2026 College of Natural Sciences Research Festival research project that motivated this work.


========================================================================
