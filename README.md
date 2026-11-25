# 🏥 Medical Insurance Cost Prediction using Multiple Linear Regression

[![GitHub Profile](https://img.shields.io/badge/GitHub-complexiota-181717?style=flat&logo=github)](https://github.com/complexiota)
[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end econometric and statistical machine learning project implementing **Multiple Linear Regression (Ordinary Least Squares - OLS)** to predict individual medical healthcare charges and identify key cost drivers.

Maintained by **[@complexiota](https://github.com/complexiota)**.

---

## 📌 Problem Statement

Health insurance providers, actuarial analysts, and healthcare networks need to understand which demographic and lifestyle biomarkers drive up medical expenses to:
- **Accurately price insurance policies** based on quantifiable empirical risk.
- **Design targeted preventative health interventions** for vulnerable cohorts.
- **Provide transparent cost estimates** for patients and hospital systems.

---

## 🎯 Project Objectives

1. **Model Building**: Train a Multiple Linear Regression model on patient demographics and lifestyle factors.
2. **Feature Engineering**: Capture the non-linear interaction effect between smoking status and body mass index (`smoker × BMI`).
3. **Statistical Inference**: Estimate standardized coefficients ($\beta$) to determine exact feature importance.
4. **OLS Assumption Verification**: Rigorously validate the 5 classical Gauss-Markov assumptions (Linearity, Normality of Residuals, Homoscedasticity, Multicollinearity via VIF, and Independence via Durbin-Watson).
5. **Real-World Forecasting**: Deploy the trained pipeline to predict annual charges for new patient profiles.

---

## 📊 Dataset Overview

- **Source**: [Medical Cost Personal Dataset (Kaggle: `mirichoi0218/insurance`)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Observations**: 1,338 individual patient records
- **Missing Values**: 0 (Clean dataset)
- **Target Variable**: `charges` (Annual medical costs billed to health insurance in USD)

| Feature | Type | Description |
|---|---|---|
| `age` | Integer | Age of primary beneficiary (18–64) |
| `sex` | Categorical | Insurance contractor gender (`female`, `male`) |
| `bmi` | Float | Body Mass Index ($kg/m^2$), ideal range: 18.5 to 24.9 |
| `children` | Integer | Number of children/dependents covered by health insurance |
| `smoker` | Categorical | Smoking status (`yes`, `no`) |
| `region` | Categorical | Beneficiary's residential area in the US (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges` | Float | **Target**: Individual medical costs billed by health insurance |

---

## 📈 Key Results & Metrics

| Metric | Score | Interpretation |
|---|---|---|
| **$R^2$ Score** | **0.8653** | Model explains **~86.5%** of variance in medical charges |
| **Adjusted $R^2$ Score** | **0.8605** | Penalized for feature count; demonstrates no overfitting |
| **Mean Absolute Error (MAE)** | **\$2,756.90** | Average prediction deviation from actual charges |
| **Root Mean Squared Error (RMSE)** | **\$4,573.81** | Standard deviation of the unexplained variance |
| **Mean Squared Error (MSE)** | **\$20,919,716.75** | Quadratic penalty metric |

---

## 🧬 Feature Importance (Standardized Coefficients)

| Rank | Feature | Standardized $\beta$ | Impact Description |
|:---:|---|:---:|---|
| 1 | **`smoker_bmi`** (Interaction) | **+18,608.73** | Exponentially amplifies charges for obese smokers |
| 2 | **`age`** | **+3,705.22** | Medical costs predictably climb with age |
| 3 | **`children`** | **+563.53** | Marginal cost increase per dependent |
| 4 | **`bmi`** | **+122.33** | Baseline positive cost pressure from BMI |
| 5 | `sex_male` | -262.54 | Slight variance between genders |
| 6 | `region_northwest` | -269.38 | Minor geographic pricing difference |
| 7 | `region_southeast` | -426.72 | Minor geographic pricing difference |
| 8 | `region_southwest` | -531.74 | Minor geographic pricing difference |
| 9 | `smoker_yes` | -8,570.66 | Base adjustment term absorbed by interaction |

> **Key Clinical Insight**: Smoking alone is dangerous, but **obesity combined with smoking (`smoker × BMI`)** is the primary driver of catastrophic healthcare expenditure ($r = 0.8451$ with charges).

---

## 🔍 Validation of OLS Regression Assumptions

| Assumption | Diagnostic Method | Test Statistic | Result / Conclusion |
|---|---|---|---|
| **1. Linearity** | Pairplot & Residual vs. Fitted | Visual inspection | ✅ Linear relationship satisfied |
| **2. Normality of Residuals** | Shapiro-Wilk Test & Q-Q Plot | $W = 0.6265, p < 0.001$ | ⚠️ Mild right-skew in extreme outliers (acceptable for $N > 1000$) |
| **3. Homoscedasticity** | Residuals vs. Fitted plot | Visual inspection | ✅ Random scatter around zero line |
| **4. No Multicollinearity** | Variance Inflation Factor (VIF) | Predictor VIFs $< 5$ | ✅ Base features show no multicollinearity (`smoker_bmi` shares variance by design) |
| **5. Independence of Errors**| Durbin-Watson Test | **$d = 2.1548$** | ✅ No autocorrelation ($1.5 < d < 2.5$) |

---

## 🩺 Sample Inference: New Patient Prediction

For a hypothetical new patient profile:
- **Age**: 35 years old
- **BMI**: 28.5 (Overweight)
- **Children**: 2
- **Gender**: Male
- **Smoker**: Yes
- **Region**: Northeast

```text
==================================================
  New Patient Prediction
==================================================
  Predicted Annual Medical Charge: $28,868.91
==================================================
```

---

## 📁 Repository Structure

```text
MultiLinearRegressionProject/
├── Multi_Linear_Regression_Model.ipynb   # Interactive 15-step Jupyter Notebook
├── README.md                             # Comprehensive project documentation
├── output/                               # Generated diagnostic & EDA plots
│   ├── 01_charges_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_pairplot.png
│   ├── 04_categorical_boxplots.png
│   ├── 05_actual_vs_predicted.png
│   ├── 06_residual_analysis.png
│   └── 07_feature_importance.png
└── .venv/                                # Local virtual environment
```

---

## 🚀 How to Run

### Option 1: Run via VS Code or JupyterLab
1. Clone or open the repository.
2. Select the virtual environment kernel (`.venv`).
3. Open `Multi_Linear_Regression_Model.ipynb` and click **Run All**.

### Option 2: Run via Terminal
```powershell
cd "e:\Third Year\R&C Algorithm\Projects\MultiLinearRegressionProject"
.\.venv\Scripts\activate
python -c "
import kagglehub, os, pandas as pd
path = kagglehub.dataset_download('mirichoi0218/insurance')
print('Dataset ready at:', path)
"
```

---

## 👤 Author & Connect

Developed by **[complexiota](https://github.com/complexiota)**.

- **GitHub Profile**: [https://github.com/complexiota](https://github.com/complexiota)
- Contributions, issues, and feature suggestions are welcome!
