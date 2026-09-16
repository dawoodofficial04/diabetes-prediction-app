# Diabetes Prediction App

A Streamlit web app that predicts the likelihood of diabetes from basic health metrics, using a Logistic Regression model trained on the Pima Indians Diabetes dataset.

**Live App:** <https://diabetes-prediction-app-gg66tcnmaqmf3yubxzjxxr.streamlit.app>

> ⚠️ **Disclaimer:** This tool is a machine learning demo, not a medical device. It flags risk based on patterns in historical data and should never be used as a substitute for professional medical advice or diagnosis.

## Overview

Enter number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age, and the app returns a Diabetic / Non-Diabetic prediction with a confidence score.

## Model

- **Algorithm:** Logistic Regression (`C=0.5`, `class_weight='balanced'`)
- **Pipeline:** median imputation → feature scaling → logistic regression, bundled into a single `scikit-learn` `Pipeline` so the saved model handles preprocessing automatically
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

### Test set performance (stratified 80/20 split)

| Metric | Before | After |
|---|---|---|
| Train / Test accuracy | 0.79 / 0.71 | 0.76 / 0.73 |
| Train–test gap (overfitting signal) | 0.084 | **0.027** |
| Recall — diabetic class | 0.52 | **0.70** |
| Precision — diabetic class | 0.60 | 0.60 |
| F1 — diabetic class | 0.55 | 0.65 |
| Mean 5-fold CV accuracy | ~0.77 | ~0.75–0.77 |

Overall accuracy stayed about the same, but **recall on the diabetic class jumped from 0.52 to 0.70** — meaning the improved model catches far more actual diabetic cases. In a health-screening context, missing a real case is a worse failure than an extra false alarm, so this is the metric that matters most here.

## Why this dataset caps out where it does

The Pima Indians Diabetes dataset has only 768 rows and 8 features, with no data on diet, family history detail, physical activity, or genetics beyond a single pedigree score. Combined with class imbalance (500 non-diabetic vs 268 diabetic cases), there's a real ceiling on how much any model can extract from it. Random Forest, Gradient Boosting, and SVM were all tested and performed no better than tuned logistic regression, confirming the limit is in the data, not the algorithm.

## What was fixed / improved from the original baseline

1. **Hidden missing data:** Glucose, BloodPressure, SkinThickness, Insulin, and BMI used `0` to mean "not recorded" — a value that's medically impossible for a living person (e.g. nearly half of all Insulin values were `0`). These were converted to missing values and filled in with the column median instead of being treated as real zeros.
2. **Feature scaling:** added `StandardScaler`, since logistic regression's regularization (`C`) assumes features are on comparable scales — Glucose (~100s) and DiabetesPedigreeFunction (~0–1) weren't.
3. **Class imbalance:** added `class_weight='balanced'` to correct for the 500-vs-268 class split, which was quietly suppressing recall on the diabetic class.
4. **Evaluation:** switched to a stratified train/test split and stratified 5-fold cross-validation, and added an explicit train-vs-test accuracy check so overfitting is visible at a glance.

## Project structure

```
diabetes_app/
├── app.py                        # Streamlit app
├── logistic_regression.ipynb     # Data prep, training, and evaluation
├── diabetes.csv                  # Dataset
├── logistic_regression_model.pkl # Trained pipeline (imputer + scaler + classifier)
├── requirements.txt
└── README.md
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If you retrain the model (re-run `logistic_regression.ipynb`), it will overwrite `logistic_regression_model.pkl` — restart the Streamlit app afterward to pick up the new file.

## Tech Stack

- **[Python 3](https://www.python.org/)**
- **[pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/)** — data loading and preprocessing
- **[scikit-learn](https://scikit-learn.org/)** — Logistic Regression, `Pipeline`, `SimpleImputer`, `StandardScaler`, cross-validation
- **[Streamlit](https://streamlit.io/)** — web app UI
- **[Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/)** — EDA and confusion matrix plot (notebook only)
- **[Jupyter Notebook](https://jupyter.org/)** — model training and experimentation
- **[Pickle](https://docs.python.org/3/library/pickle.html)** — model serialization

## Dataset

[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — 768 rows, 9 columns (8 features + Outcome).

## Limitations & possible next steps

- Small dataset (768 rows) with no lifestyle, diet, or detailed family history features
- Class imbalance (65% non-diabetic / 35% diabetic) limits how far precision and recall can be pushed together
- More data or richer features (activity level, diet, HbA1c) would likely help more than further model tuning
- The app is a demo/learning project only — not validated for clinical use