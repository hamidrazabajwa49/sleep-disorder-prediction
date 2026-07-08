<div align="center">

# 😴 Sleep Disorder Prediction

### Two classification tasks, one lifestyle & health dataset

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-EB5E28?style=flat-square)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-4CAF50?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

**374 individuals · lifestyle, physiological & occupational features**

</div>

---

## 📌 TL;DR

> Two related classification tasks on the same 374-row dataset: **(A)** does this person have *any* sleep disorder, and **(B)** *which* one — Insomnia or Sleep Apnea? Both tasks hit very high scores across all models. That's a property of this dataset (see [§ Honest Take](#-honest-take)), not a claim of clinical-grade prediction.

<div align="center">

| Task | Best Model | Key Metric |
|:---|:---|:---:|
| **A — Binary** (disorder / no disorder) | Random Forest / XGBoost (tie) | ROC-AUC 0.986 / 0.991, F1 0.984 |
| **B — Multiclass** (None / Insomnia / Apnea) | Logistic Regression | ROC-AUC 0.948, F1-macro 0.845 |

</div>

---

## 🗂️ Table of Contents

- [Dataset](#-dataset)
- [Cleaning](#-cleaning)
- [Task A — Binary](#-task-a--binary-classification)
- [Task B — Multiclass](#-task-b--multiclass-classification)
- [Feature Importance](#-feature-importance)
- [Honest Take](#-honest-take)
- [Project Structure](#-project-structure)
- [Quickstart](#-quickstart)

---

## 📊 Dataset

<details>
<summary><b>374 rows × 13 raw columns — click to expand</b></summary>

| Attribute | Value |
|---|---|
| Rows | 374 |
| Sleep Disorder | None 219 (59%) · Sleep Apnea 78 (21%) · Insomnia 77 (21%) |
| Features | Age, Gender, Occupation, Sleep Duration, Quality of Sleep, Physical Activity Level, Stress Level, BMI Category, Blood Pressure, Heart Rate, Daily Steps |

</details>

## 🧹 Cleaning

| Issue Found | Fix |
|---|---|
| `Sleep Disorder` is `NaN` for people with no disorder | Filled explicitly with `'None'` |
| `BMI Category` has duplicate labels (`Normal` / `Normal Weight`) | Consolidated to `Normal` |
| `Blood Pressure` stored as a string (`"126/83"`) | Split into `BP Systolic` / `BP Diastolic` integer columns |
| Sparse `Occupation` categories | Software Engineer, Scientist, Sales Rep, Manager → grouped as `Other` before one-hot encoding |

## 🎯 Task A — Binary Classification

**Question:** does this person have any sleep disorder?

<div align="center">

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.886 | 1.000 | 0.939 | 0.994 | 0.990 |
| **Random Forest** | **0.969** | 1.000 | **0.984** | 0.986 | 0.970 |
| **XGBoost** | **0.969** | 1.000 | **0.984** | **0.991** | 0.984 |

</div>

<details>
<summary><b>Threshold tuning (XGBoost)</b></summary>

Swept 0.10 → 0.90 in steps of 0.01 to maximize F1.

```
Best threshold : 0.43
Precision      : 0.969
Recall         : 1.000
F1             : 0.984
```

Barely moves the needle from the default 0.5 threshold — the model is already well-calibrated for this task.

</details>

<details>
<summary><b>5-fold cross-validation (XGBoost, train set only)</b></summary>

```
ROC-AUC: 0.921 ± 0.038   (folds: 0.901, 0.947, 0.929, 0.967, 0.858)
PR-AUC : 0.885 ± 0.026   (folds: 0.861, 0.881, 0.905, 0.922, 0.855)
```

CV scores sit noticeably below the single-split test scores above — a useful reminder that a single 75-row test split (this dataset's full holdout) is a shaky ground for point-estimate bragging rights. The CV spread (fold range 0.86–0.97) is the more honest picture.

</details>

## 🎯 Task B — Multiclass Classification

**Question:** which disorder — None, Insomnia, or Sleep Apnea?

<div align="center">

| Model | F1 Macro | F1 Weighted | ROC-AUC (macro OvR) |
|---|:---:|:---:|:---:|
| **Logistic Regression** | 0.845 | 0.875 | **0.948** |
| Random Forest | **0.856** | **0.897** | 0.940 |
| XGBoost | 0.823 | 0.872 | 0.930 |

</div>

**Per-class breakdown (Logistic Regression):**

| Class | Precision | Recall | F1 |
|---|:---:|:---:|:---:|
| None | 1.00 | 0.86 | 0.93 |
| Insomnia | 0.92 | 0.80 | 0.86 |
| Sleep Apnea | 0.62 | 0.94 | 0.75 |

**Sleep Apnea is the weak spot** — high recall but the lowest precision of the three classes across every model tested, meaning the models over-predict Sleep Apnea and pull some Insomnia/None cases into it.

<details>
<summary><b>5-fold cross-validation (XGBoost, train set only)</b></summary>

```
F1 Macro         : 0.886 ± 0.041   (folds: 0.881, 0.860, 0.918, 0.827, 0.942)
ROC-AUC (OvR wt) : 0.907 ± 0.048   (folds: 0.870, 0.836, 0.969, 0.923, 0.939)
```

</details>

## 🔍 Feature Importance

Per-class ROC curves and XGBoost gain-based importances are in the notebook (§8). Occupation, BMI category, stress level, and the derived blood-pressure columns dominate both tasks' top-15 feature lists.

## 🤔 Honest Take

- **These scores are unusually high for a health classification task** — ROC-AUC in the 0.95–0.99 range on the binary task is not typical of real clinical prediction problems. This is a well-documented characteristic of this specific Kaggle dataset: the relationship between BMI, blood pressure, stress, and disorder label is close to rule-based rather than noisy, real-world clinical signal. Treat these numbers as *"how well can a model recover a near-deterministic pattern in synthetic-leaning data,"* not *"how well can ML diagnose sleep disorders."*
- **The 75-row test set is small.** Cross-validation scores (which use more folds over the training data) are meaningfully lower and more variable than the single test-split numbers — the CV numbers are the more trustworthy estimate of generalization.
- **Sleep Apnea precision is the consistent weak point** across every model in the multiclass task, worth digging into further if this were a production model rather than a portfolio project.

## 📁 Project Structure

```
sleep-disorder-prediction/
├── data/
│   └── Sleep_health_and_lifestyle_dataset.csv
├── sleep_disorder_prediction.ipynb
├── README.md
└── requirements.txt
```

## 🚀 Quickstart

```bash
git clone <this-repo>
cd sleep-disorder-prediction
pip install -r requirements.txt
jupyter notebook sleep_disorder_prediction.ipynb
```

---

<div align="center">

**Stack:** Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn

</div>
