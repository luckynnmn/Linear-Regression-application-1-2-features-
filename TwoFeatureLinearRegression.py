from __future__ import annotations

"""
Take-home assignment: two-feature linear regression (bmi + age -> expenses).

    expenses = w0 + w1 * bmi + w2 * age

Trains the same model two ways (pure Python, no numpy -- same style as
Module 2 scripts 01-04):
  1) Normal Equation with a GENERAL linear solver (Gauss-Jordan elimination),
     not the 2x2 shortcut from 02_ols_normal_equation.py.
  2) Gradient Descent on standardized features (same pattern as
     03_gradient_descent.py), converting weights back to original units.

Compares both against the bmi-only baseline (02_ols_normal_equation.py),
prints a comparison table, and writes reports/assignment_results.csv.
"""

import csv
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# If the script cannot find insurance.csv on its own (e.g. when pasted into
# a Jupyter cell on your own machine), put the full path to your CSV here.
# Leave it as "" to use the default location (next to this script, or the
# notebook's working directory when pasted into a cell).
# NOTE for submission: clear this back to "" so the grader's layout is used.
CSV_PATH_OVERRIDE = ""
try:
    _BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # Pasted into a Jupyter notebook cell: __file__ is not defined there,
    # so fall back to the notebook's working directory.
    _BASE_DIR = Path.cwd()

CSV_PATH = Path(CSV_PATH_OVERRIDE) if CSV_PATH_OVERRIDE else _BASE_DIR / "data" / "insurance-premium-prediction" / "insurance.csv"
REPORT_PATH = _BASE_DIR / "reports" / "assignment_results.csv"
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Could not find the dataset at: {CSV_PATH}\n"
        "Set CSV_PATH_OVERRIDE at the top of the script/cell to the full path "
        "of your insurance.csv."
    )

FEATURES = ["bmi", "age"]
TARGET = "expenses"

LEARNING_RATE = 0.05
EPOCHS = 10000


# ---------------------------------------------------------------------------
# Data loading / helpers (same pattern as Module 2 scripts)
# ---------------------------------------------------------------------------
def load_columns(csv_path: Path, features: list[str], target: str):
    xs: list[list[float]] = [[] for _ in features]
    y: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for i, feat in enumerate(features):
                xs[i].append(float(row[feat]))
            y.append(float(row[target]))
    return xs, y


def standardize(values: list[float]) -> tuple[list[float], float, float]:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std == 0:
        return [0.0 for _ in values], mean, 1.0
    return [(v - mean) / std for v in values], mean, std


def mse(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    y_mean = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    return 1.0 - (ss_res / ss_tot if ss_tot else 0.0)


# ---------------------------------------------------------------------------
# Method 1: Normal Equation with a general linear solver (Gauss-Jordan)
# ---------------------------------------------------------------------------
def solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b via Gauss-Jordan elimination with partial pivoting."""
    n = len(a)
    # Augmented matrix [A | b].
    m = [list(a[i]) + [b[i]] for i in range(n)]

    for col in range(n):
        # Partial pivot: row with largest |value| in this column.
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix: cannot solve normal equation.")
        m[col], m[pivot] = m[pivot], m[col]

        # Normalize the pivot row.
        piv = m[col][col]
        m[col] = [v / piv for v in m[col]]

        # Eliminate this column from every other row.
        for r in range(n):
            if r != col:
                factor = m[r][col]
                if factor != 0.0:
                    m[r] = [rv - factor * cv for rv, cv in zip(m[r], m[col])]

    return [m[i][n] for i in range(n)]


def fit_normal_equation(feature_cols: list[list[float]], y: list[float]) -> list[float]:
    """Fit y = w0 + w1*x1 + ... via theta = (X^T X)^-1 X^T y (general case)."""
    n = len(y)
    cols = [[1.0] * n] + feature_cols  # first column = intercept
    p = len(cols)

    xtx = [
        [sum(ca * cb for ca, cb in zip(cols[i], cols[j])) for j in range(p)]
        for i in range(p)
    ]
    xty = [sum(c * yt for c, yt in zip(col, y)) for col in cols]

    return solve_linear_system(xtx, xty)


# ---------------------------------------------------------------------------
# Method 2: Gradient Descent on standardized features
# ---------------------------------------------------------------------------
def fit_gradient_descent(
    feature_cols: list[list[float]],
    y: list[float],
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
) -> tuple[list[float], list[tuple[int, float]]]:
    """Full-batch GD on standardized features; weights returned in original units."""
    stats = [standardize(col) for col in feature_cols]  # (z_values, mean, std)
    z_cols = [s[0] for s in stats]
    n = len(y)
    p = len(z_cols)

    w = [0.0] * (p + 1)  # w[0] = intercept in standardized space
    trace: list[tuple[int, float]] = []

    for epoch in range(1, epochs + 1):
        preds = [w[0] + sum(w[j + 1] * z_cols[j][i] for j in range(p)) for i in range(n)]
        errors = [pr - yt for pr, yt in zip(preds, y)]

        w[0] -= learning_rate * (2.0 / n) * sum(errors)
        for j in range(p):
            grad = (2.0 / n) * sum(e * z for e, z in zip(errors, z_cols[j]))
            w[j + 1] -= learning_rate * grad

        if epoch == 1 or epoch % 1000 == 0:
            trace.append((epoch, mse(y, preds)))

    # Convert weights back to original feature units:
    #   y = a0 + a1*z1 + a2*z2,  zj = (xj - mean_j)/std_j
    #   => w_j = a_j / std_j,  w0 = a0 - sum(a_j * mean_j / std_j)
    w_orig = [0.0] * (p + 1)
    for j in range(p):
        _, mean_j, std_j = stats[j]
        w_orig[j + 1] = w[j + 1] / std_j
    w_orig[0] = w[0] - sum(w[j + 1] * stats[j][1] / stats[j][2] for j in range(p))
    return w_orig, trace


def predict(feature_cols: list[list[float]], weights: list[float]) -> list[float]:
    n = len(feature_cols[0])
    p = len(feature_cols)
    return [weights[0] + sum(weights[j + 1] * feature_cols[j][i] for j in range(p)) for i in range(n)]


def evaluate(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Take-home assignment: two-feature model (bmi + age -> expenses)")
    print(f"Dataset: {CSV_PATH}")
    print()

    (bmi, age), expenses = load_columns(CSV_PATH, FEATURES, TARGET)
    print(f"Loaded {len(expenses)} rows.")
    print()

    # --- Model A: two-feature Normal Equation (general solver) ---
    w_ne = fit_normal_equation([bmi, age], expenses)
    pred_ne = predict([bmi, age], w_ne)
    eval_ne = evaluate(expenses, pred_ne)

    # --- Model B: two-feature Gradient Descent (standardized) ---
    w_gd, gd_trace = fit_gradient_descent([bmi, age], expenses)
    pred_gd = predict([bmi, age], w_gd)
    eval_gd = evaluate(expenses, pred_gd)

    # --- Baseline: bmi-only Normal Equation (script 02 style) ---
    w_base = fit_normal_equation([bmi], expenses)
    pred_base = predict([bmi], w_base)
    eval_base = evaluate(expenses, pred_base)

    print("Normal Equation (two features):")
    print(f"  expenses = {w_ne[0]:.4f} + {w_ne[1]:.4f}*bmi + {w_ne[2]:.4f}*age")
    print()
    print("Gradient Descent (two features, standardized then converted back):")
    print(f"  expenses = {w_gd[0]:.4f} + {w_gd[1]:.4f}*bmi + {w_gd[2]:.4f}*age")
    print(f"  hyperparameters: learning_rate={LEARNING_RATE}, epochs={EPOCHS}")
    print(f"  first/last checkpoint MSE: {gd_trace[0][1]:.2f} -> {gd_trace[-1][1]:.2f}")
    print()
    print("Baseline (bmi only, Normal Equation):")
    print(f"  expenses = {w_base[0]:.4f} + {w_base[1]:.4f}*bmi")
    print()

    rows = [
        ("normal_equation_2feat", w_ne[0], w_ne[1], w_ne[2], eval_ne),
        ("gradient_descent_2feat", w_gd[0], w_gd[1], w_gd[2], eval_gd),
        ("baseline_bmi_only", w_base[0], w_base[1], float("nan"), eval_base),
    ]

    header = f"{'model':<24} {'w0':>12} {'w_bmi':>12} {'w_age':>12} {'MSE':>14} {'RMSE':>12} {'MAE':>12} {'R^2':>8}"
    print("Comparison table")
    print(header)
    print("-" * len(header))
    for name, w0, w1, w2, ev in rows:
        w2_str = f"{w2:>12.4f}" if not math.isnan(w2) else f"{'n/a':>12}"
        print(
            f"{name:<24} {w0:>12.4f} {w1:>12.4f} {w2_str}"
            f" {ev['mse']:>14.2f} {ev['rmse']:>12.2f} {ev['mae']:>12.2f} {ev['r2']:>8.4f}"
        )
    print()

    r2_gain = eval_ne["r2"] - eval_base["r2"]
    print(f"R^2 gain from adding age (Normal Equation): {r2_gain:.4f}")
    w_diff = [abs(a - b) for a, b in zip(w_ne, w_gd)]
    print(f"|w_NE - w_GD| per coefficient: {[f'{d:.6f}' for d in w_diff]}")
    print()

    # --- Write CSV report ---
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "w0", "w_bmi", "w_age", "mse", "rmse", "mae", "r2"])
        for name, w0, w1, w2, ev in rows:
            writer.writerow([
                name,
                f"{w0:.6f}",
                f"{w1:.6f}",
                "" if math.isnan(w2) else f"{w2:.6f}",
                f"{ev['mse']:.4f}",
                f"{ev['rmse']:.4f}",
                f"{ev['mae']:.4f}",
                f"{ev['r2']:.6f}",
            ])
    print(f"Wrote comparison report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
