# Breast Cancer Classification

A neural-network classifier, built with TensorFlow/Keras, that predicts whether a breast tumour sample is malignant or benign from 30 numeric features of the Breast Cancer Wisconsin (Diagnostic) dataset. It is a tabular binary classification project built for learning purposes.

## Problem Statement
Given 30 numeric measurements computed from digitised images of fine-needle-aspirate (FNA) samples of breast masses, classify each sample as malignant or benign.

## Objectives
- Explore and preprocess the Breast Cancer Wisconsin Diagnostic dataset.
- Build a neural-network-based binary classification model to classify samples as malignant or benign.
- Apply appropriate feature scaling and stratified train-test splitting to prepare the data for modelling.
- Evaluate model performance using accuracy, precision, recall, F1-score, ROC-AUC and a confusion matrix.

## Dataset
The Breast Cancer Wisconsin (Diagnostic) dataset, loaded through `sklearn.datasets.load_breast_cancer`. Because it is provided by scikit-learn, the repository intentionally contains no dataset file.

- 569 samples, 30 numeric features (mean, standard error and "worst" value of ten cell-nucleus measurements)
- Binary target: 212 malignant and 357 benign samples
- No missing values and no duplicate rows

Source: UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic) dataset.

Target mapping (verified in the notebook against the dataset's `target_names`):

| Value | Class |
|-------|-------|
| 0 | Malignant |
| 1 | Benign |

## Methodology
1. Load the dataset and verify the target mapping.
2. Explore the data and run quality checks (missing values, duplicates, feature types).
3. Split into stratified 80/20 train and test sets with `random_state=42` (455 training and 114 test samples).
4. Fit `StandardScaler` on the training data only, then transform the training and test data.
5. Train the neural network, using 10% of the training data for validation.
6. Evaluate once on the held-out test set.
7. Run an example prediction using the same fitted scaler and trained model.

## Model
- Input: 30 standardised features
- Hidden layer: Dense(20, ReLU)
- Output layer: Dense(2, Softmax)
- Optimizer: Adam
- Loss: Sparse categorical cross-entropy
- Epochs: 10

## Results
Evaluated on the held-out test set of 114 samples. ROC-AUC is computed from the predicted probability of class 1 (Benign).

| Metric | Value |
|--------|-------|
| Test accuracy | 0.9474 (94.74%) |
| ROC-AUC | 0.9821 |

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Malignant | 0.9286 | 0.9286 | 0.9286 | 42 |
| Benign | 0.9583 | 0.9583 | 0.9583 | 72 |

Confusion matrix (rows = actual class, columns = predicted class):

| | Predicted Malignant | Predicted Benign |
|---|---|---|
| **Actual Malignant** | 39 | 3 |
| **Actual Benign** | 3 | 69 |

These results come from a single train/test split on a benchmark dataset and do not represent clinical performance.

## Example Prediction
The notebook classifies one sample, given as a dictionary of all 30 named features, using the scaler fitted on the training data. The sample is a record from the dataset itself, so this demonstrates the inference workflow only; it is not independent validation.

```
Predicted class: Benign
Class probabilities: Malignant = 0.0422, Benign = 0.9578
```

## Technologies Used
Python, NumPy, pandas, Matplotlib, scikit-learn, TensorFlow/Keras, Jupyter.

## Project Structure
```
breast-cancer-classification-ml/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── breast_cancer_classification.ipynb
```

## How to Run
Tested with Python 3.13.15 and the package versions pinned in `requirements.txt`.

**Locally**
```bash
pip install -r requirements.txt
pip install jupyter
jupyter notebook notebooks/breast_cancer_classification.ipynb
```

**Google Colab**: upload `notebooks/breast_cancer_classification.ipynb` and choose Runtime → Run all (the required libraries are preinstalled).

Random seeds are fixed (`RANDOM_STATE = 42`), but results may differ slightly across hardware and library versions.

## Limitations
- The dataset contains 569 samples, so the evaluation is based on a relatively small benchmark dataset.
- Evaluation uses a single stratified train-test split rather than cross-validation.
- The neural-network architecture and training settings were kept simple and were not extensively tuned.
- The model has not been externally or clinically validated, and performance may not generalise to other datasets, populations or clinical settings.

## Disclaimer
This project is for educational purposes only. It is not a medical device and must not be used for diagnosis, treatment or clinical decisions. Medical concerns should be addressed by qualified healthcare professionals.
