# Linear Regression Application (1 Feature and 2 Features)

Predicting medical insurance `expenses` from customer data using linear regression, built from scratch in pure Python (no numpy).

The project starts with a **1-feature model** (`bmi -> expenses`) and extends it to a **2-feature model** (`bmi + age -> expenses`), because BMI alone explains very little of the variation in expenses (R² ≈ 0.04).

## Results

| Model | Equation | RMSE | MAE | R² |
|---|---|---|---|---|
| 1 feature (BMI only, Normal Equation) | `expenses = 1178.1795 + 394.3276*bmi` | 11,864.41 | 9,172.30 | 0.0394 |
| 2 features (BMI + age, Normal Equation) | `expenses = -6437.3475 + 333.3909*bmi + 241.9001*age` | 11,373.64 | 9,032.28 | 0.1173 |
| 2 features (BMI + age, Gradient Descent) | same coefficients as Normal Equation | 11,373.64 | 9,032.28 | 0.1173 |

Adding `age` raises R² by about **0.078**, and Gradient Descent converges to the same coefficients as the Normal Equation. R² is still low, so most of the variation in expenses is driven by variables not included in this model, which is a natural next step.

## Where to find each model

| What | File |
|---|---|
| **1 feature** (BMI): simple linear regression | [`01_simple_linear.py`](01_simple_linear.py) |
| **1 feature** (BMI): Normal Equation (OLS) | [`02_ols_normal_equation.py`](02_ols_normal_equation.py) |
| **1 feature** (BMI): Gradient Descent | [`03_gradient_descent.py`](03_gradient_descent.py) |
| **1 feature** (BMI): model comparison with visual | [`04_compare_models_visual.py`](04_compare_models_visual.py) |
| **2 features** (BMI + age): Normal Equation and Gradient Descent, compared with the 1-feature baseline | [`TwoFeatureLinearRegression.py`](TwoFeatureLinearRegression.py) |
| Jupyter notebook 1 feature lab | [`linear_regression_lab.ipynb`](linear_regression_lab.ipynb) |
| Comparison results (written by the 2-feature script) | [`reports/assignment_results.csv`](reports/assignment_results.csv) |
| Dataset | [`data/insurance-premium-prediction/insurance.csv`](data/insurance-premium-prediction/insurance.csv) |

## Repository structure

```
.
├── 01_simple_linear.py
├── 02_ols_normal_equation.py
├── 03_gradient_descent.py
├── 04_compare_models_visual.py
├── TwoFeatureLinearRegression.py
├── linear_regression_lab.ipynb
├── requirements.txt
├── data/
│   ├── insurance-premium-prediction/
│   │   └── insurance.csv
│   └── model_comparison_bmi_expenses.png
└── reports/
    └── assignment_results.csv
```

## How the 2-feature model works

`TwoFeatureLinearRegression.py` trains `expenses = w0 + w1*bmi + w2*age` two ways:

1. **Normal Equation** with a general linear solver (Gauss-Jordan elimination), not the 2x2 shortcut used in the 1-feature script.
2. **Gradient Descent** on standardized features (learning rate 0.05, 10,000 epochs), then converts the weights back to the original units.

It compares both against the BMI-only baseline, prints a comparison table (MSE, RMSE, MAE, R²), and writes `reports/assignment_results.csv`.

## Getting started

### Requirements

- [Git](https://git-scm.com/downloads)
- Python 3.8 or newer

### 1. Clone the repository

The command is the same on Windows and macOS:

```bash
git clone https://github.com/luckynnmn/Linear-Regression-application-1-2-features-.git
cd Linear-Regression-application-1-2-features-
```

No Git? On the GitHub page click **Code > Download ZIP**, unzip, and open the folder in a terminal.

### 2. Create a virtual environment and install packages

**Windows (PowerShell):**

```powershell
python -m venv LinearRegression
.\LinearRegression\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run this once in the same window and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

**macOS (Terminal):**

```bash
python3 -m venv LinearRegression
source LinearRegression/bin/activate
pip install -r requirements.txt
```

### 3. Run the models

**1 feature (BMI):**

```bash
python 01_simple_linear.py
python 02_ols_normal_equation.py
python 03_gradient_descent.py
python 04_compare_models_visual.py
```

**2 features (BMI + age):**

```bash
python TwoFeatureLinearRegression.py
```

On macOS use `python3` instead of `python` if `python` is not found (inside the activated virtual environment, `python` also works).

`TwoFeatureLinearRegression.py` uses only the Python standard library, so it runs even without installing `requirements.txt`. The packages are needed for the plotting script and the notebook.

### 4. Open the notebook (optional)

```bash
jupyter notebook linear_regression_lab.ipynb
```

If Jupyter is not installed: `pip install notebook`.

## Contact

Nhi (Lucky) Nguyen — [@luckynnmn](https://github.com/luckynnmn)
