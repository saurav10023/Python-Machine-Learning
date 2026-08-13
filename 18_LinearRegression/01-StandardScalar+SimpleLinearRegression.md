# StandardScaler + Simple Linear Regression — Complete Notes

---

## 1. What is `StandardScaler`?

A preprocessing tool from `sklearn.preprocessing` that **standardizes** features so each column has:
- **Mean (μ) = 0**
- **Standard deviation (σ) = 1**

### Formula (applied to every value, per column)

```
z = (x - μ) / σ
```

| Symbol | Meaning |
|---|---|
| x | original value |
| μ | mean of that column |
| σ | standard deviation of that column |
| z | standardized (scaled) value |

### What it does NOT do
- It does **not** bound values into a fixed range like [-1, 1] or [0, 1].
- That's a different tool: `MinMaxScaler`.
- After standardizing, most values fall roughly between -3 and 3 (for normally distributed data), but there's no hard cap — outliers can go further.

### Key property: it's a linear transformation
- Only **shifts** (centers at 0) and **stretches/compresses** (rescales spread to 1) the data.
- The **order** and **relative spacing** of data points is preserved.
- The **shape** of the distribution doesn't change (skewed stays skewed, normal stays normal).

---

## 2. Why do we use it?

Many ML algorithms compute **distances**, **dot products**, or **gradients** directly from raw feature values. If one feature has a much larger numeric range than another, it will **dominate** the calculation — not because it's more important, but purely because its numbers are bigger.

**Example:**
- `age`: 18–90
- `income`: 20,000–500,000

Without scaling, `income` swamps `age` in any distance-based or gradient-based computation.

### Algorithms that need/benefit from scaling
- Logistic Regression, Linear Regression (esp. with Ridge/Lasso regularization)
- KNN, K-Means (distance-based)
- SVM
- PCA
- Neural Networks (helps gradient descent converge faster)

### Algorithms that generally DON'T need it
- Decision Trees, Random Forests, Gradient Boosted Trees (they split on thresholds, not distances)

> Note: For **simple single-feature linear regression**, scaling doesn't change the model's actual predictions — LR is scale-invariant with one feature. Scaling matters much more with multiple features, regularization, or distance-based algorithms.

---

## 3. `fit`, `transform`, `fit_transform` — the golden rule

```python
scalar = StandardScaler()

x_train = scalar.fit_transform(x_train)   # LEARN μ, σ from train AND apply
x_test  = scalar.transform(x_test)        # REUSE train's μ, σ — do NOT relearn
```

| Method | What it does |
|---|---|
| `fit(x_train)` | Calculates μ and σ from `x_train`, stores them inside the scaler object |
| `transform(x)` | Applies `(x - μ) / σ` using the **stored** μ, σ |
| `fit_transform(x)` | Does `fit` + `transform` in one call, on the same data |

### The rule (never break this):
**Fit only once — on training data.** Every other dataset (test, validation, new/future data) must only be `transform`-ed using the μ, σ learned from training data.

### Why? — Data Leakage
If you `fit` (or `fit_transform`) on test data, the scaler learns a **new, different** mean/std from the test set instead of reusing train's. This means:
- Information from the test set "leaks" into preprocessing.
- Train and test end up standardized on **two different, incompatible scales**.
- Your model was trained expecting one scale, but test data is now in a different scale → predictions become numerically wrong.

**Numeric example:**
- `x_train` weight column: mean = 65, std = 10
- `x_test` weight column: mean = 68, std = 8 (different sample → different stats)
- A test value of 70:
  - Correct (using train's stats): `(70-65)/10 = 0.5`
  - Wrong (using test's own stats): `(70-68)/8 = 0.25`

These feed **different numbers** into the model for the same real-world value → predictions are corrupted.

---

## 4. The original buggy code — issues found

```python
x=df[['Weight']]
...
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
...
scalar = StandardScaler()
scalar.fit_transform(x_train)          # BUG 2
x_test = scalar.fit_transform(x_test)  # BUG 3
```

### Bug 1 — `y` never defined
`y` is used in `train_test_split(x, y, ...)` but never created.
**Effect:** `NameError`. Code execution stops immediately — nothing after this line runs until fixed.

### Bug 2 — `fit_transform(x_train)` result discarded
`fit_transform` **returns** the scaled array but it's never assigned back to `x_train`.
**Effect:** No error, but `x_train` silently stays in its original **unscaled** form. Scaling never actually happens — all that work is wasted.

### Bug 3 — `fit_transform` used on `x_test` instead of `transform`
This re-fits (learns new μ, σ) on the test set instead of reusing train's stats.
**Effect:** No error, but **data leakage** — train and test end up on different scales, causing systematically wrong `y_pred` and inflated/incorrect MSE, MAE, RMSE.

### Why these are dangerous
Bugs 2 & 3 don't crash the program — they run silently and produce a working-looking but **numerically wrong** model. This is worse than a hard error because it's easy to miss.

---

## 5. Fixes — exactly what to change

| Location | Before | After |
|---|---|---|
| After `x=df[['Weight']]` | *(missing)* | Add: `y=df['Height']` |
| Scaling train | `scalar.fit_transform(x_train)` | `x_train=scalar.fit_transform(x_train)` |
| Scaling test | `x_test=scalar.fit_transform(x_test)` | `x_test=scalar.transform(x_test)` |

---

## 6. Corrected full code (line-by-line explanation)

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('height-weight.csv')      # load dataset
df.head()                                   # preview first 5 rows

# Scatter plot — visually inspect relationship between Height and Weight
plt.scatter(df['Height'], df['Weight'])
plt.xlabel("Height")
plt.ylabel("Weight")

df.corr()                                   # correlation matrix — check linear relationship strength

import seaborn as sns
sns.pairplot(df)                            # pairwise scatter plots for all numeric columns

x = df[['Weight']]                          # feature (input) — double brackets = keep as DataFrame (2D)
y = df['Height']                            # target (output) — single brackets = Series (1D)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)
# test_size=0.25 -> 25% of data reserved for testing, 75% for training
# random_state=42 -> makes the split reproducible (same split every run)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

x_train = scalar.fit_transform(x_train)     # LEARN mean & std from x_train, then scale x_train
x_test  = scalar.transform(x_test)          # REUSE the same mean & std to scale x_test (no relearning)

from sklearn.linear_model import LinearRegression
regression = LinearRegression(n_jobs=-1)    # n_jobs=-1 -> use all CPU cores
regression.fit(x_train, y_train)            # train the model: find best slope & intercept

print("Coefficient/slope:", regression.coef_)
print("Intercept:", regression.intercept_)

# Visualize training data + best fit line
plt.scatter(x_train, y_train)
plt.plot(x_train, regression.predict(x_train), 'red')

y_pred = regression.predict(x_test)         # predict on unseen (test) data

from sklearn.metrics import mean_absolute_error, mean_squared_error
mse  = mean_squared_error(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(mse)
print(mae)
print(rmse)
```

---

## 7. Evaluation metrics — quick reference

| Metric | Formula (concept) | Meaning |
|---|---|---|
| **MAE** | average of \|actual − predicted\| | average absolute error, same unit as target |
| **MSE** | average of (actual − predicted)² | penalizes larger errors more (squared) |
| **RMSE** | √MSE | brings error back to original unit; easier to interpret than MSE |

Lower values = better fit for all three.

---

## 8. StandardScaler vs MinMaxScaler — quick comparison

| Scaler | Guarantees | Typical Range |
|---|---|---|
| `StandardScaler` | mean = 0, std = 1 | Usually ~ -3 to 3, but unbounded |
| `MinMaxScaler` | fixed min/max | Exactly [0, 1] (or custom range) |

---

## 9. One-line summary for revision

> `StandardScaler` re-centers and re-scales each feature (subtract mean, divide by std) so all features are on a comparable scale for the model's math — without changing their relative order or distribution shape. **Fit only on training data; only transform on test data** — never fit on test data, or you get data leakage and a broken train/test scale mismatch.