# Logistic Regression: Complete Notes

> Covers: Binary Classification → Cost Function → Gradient Descent → Performance Metrics → Multiclass (One-vs-Rest)

**Contents**
1. [Logistic Regression (Binary Classification)](#1-logistic-regression-binary-classification)
2. [Performance Metrics](#2-performance-metrics)
3. [Multiclass: One-versus-Rest](#3-multiclass-logistic-regression-one-versus-rest)
4. [Full Python Example](#4-full-python-example-scikit-learn)
5. [Assumptions, Pros and Cons](#5-assumptions-pros-and-cons)
6. [Quick Revision Cheat Sheet](#6-quick-revision-cheat-sheet)

---

## 1. Logistic Regression (Binary Classification)

Logistic regression predicts a **category** (not a number). Despite the word "regression" in its name, it is a **classification** algorithm. It outputs a **probability between 0 and 1**, which is then converted to a class.

### 1.1 Dataset example

| Study hours (input) | Result (output) |
|---|---|
| 1 | Fail (0) |
| 2 | Fail (0) |
| 4 | Pass (1) |
| 6 | Pass (1) |

Output is **binary**: `Pass (1)` / `Fail (0)`.

### 1.2 Why can't we use Linear Regression for classification?

| Problem | Explanation |
|---|---|
| **1. Outliers** | The best-fit line shifts a lot with a single extreme point (e.g. someone who studied 20 hrs). This moves the 0.5 threshold and misclassifies other points. |
| **2. Output outside [0, 1]** | A line goes to > 1 and < 0. Probabilities can't be like that. We need to **squash** the output into [0, 1], which a straight line can't do. |

**Example:** With data at 1, 2, 3, 4, 5, 6 hours, a line fit might predict `1.4` for 7 hrs or `-0.3` for 0 hrs. These values are meaningless as probabilities.

### 1.3 How Logistic Regression solves it

Start with the same linear equation, then **squash it through a sigmoid function**:

```
z = θ₀ + θ₁x

h_θ(x) = σ(z) = 1 / (1 + e^(−z))
```

This is the **logistic regression hypothesis**.

### 1.4 Sigmoid Activation

```
σ(z) = 1 / (1 + e^(−z))
```

| z | σ(z) |
|---|---|
| very large negative (−10) | ≈ 0 |
| 0 | **0.5** |
| very large positive (+10) | ≈ 1 |

**Key properties**
- Output always lies in **(0, 1)**
- `z ≥ 0` ⟹ `σ(z) ≥ 0.5` ⟹ predict class **1**
- `z < 0` ⟹ `σ(z) < 0.5` ⟹ predict class **0**
- S-shaped curve, smooth and differentiable
- Derivative: `σ'(z) = σ(z)(1 − σ(z))`

**Interpreting the output:** `h_θ(x) = 0.8` means "80% probability that y = 1".

**Decision rule (threshold = 0.5 by default):**
```
if h_θ(x) ≥ 0.5  → predict 1
else             → predict 0
```
The threshold can be changed (see the Precision/Recall section).

### 1.5 Worked example

Suppose after training: `θ₀ = −4`, `θ₁ = 1`. For a student who studies **6 hours**:

```
z = −4 + 1(6) = 2
h(x) = 1 / (1 + e^(−2)) = 0.88  →  88% chance of Pass  →  predict Pass
```

For 2 hours: `z = −2`, `h = 0.12` → predict **Fail**.

**Decision boundary:** `z = 0` → `x = 4` hours. Below 4 hrs predict Fail, above predict Pass.

### 1.6 Cost function

**Linear regression cost (MSE):**
```
J(θ₀, θ₁) = (1/2m) Σᵢ (h_θ(xⁱ) − yⁱ)²        h_θ(x) = θ₀ + θ₁x
```
This is a **convex** function (a bowl shape) with a single global minimum.

**If we use MSE for logistic regression:**
```
h_θ(x) = 1 / (1 + e^(−z))     ← sigmoid is non-linear
```
Plugging the sigmoid inside the squared error makes J(θ) **non-convex**: many local minima, wavy surface. **Gradient descent may get stuck**, so we can't rely on it.

### 1.7 Log Loss (Binary Cross-Entropy)

The fix is a different cost function:

```
Cost(h_θ(x), y) =  −log(h_θ(x))        if y = 1
                   −log(1 − h_θ(x))    if y = 0
```

**Intuition**

| Actual y | Prediction h(x) | Cost |
|---|---|---|
| 1 | 0.99 (confident, right) | ≈ 0 (tiny) |
| 1 | 0.01 (confident, wrong) | **very large** (−log 0.01 = 4.6) |
| 0 | 0.01 (confident, right) | ≈ 0 |
| 0 | 0.99 (confident, wrong) | **very large** |

The cost **punishes confident wrong predictions heavily**.

**Combined into one equation:**
```
Cost(h_θ(x), y) = −y·log(h_θ(x)) − (1 − y)·log(1 − h_θ(x))
```
(When y = 1 the second term vanishes; when y = 0 the first vanishes.)

**Full cost function over m samples:**
```
J(θ₀, θ₁) = −(1/m) Σᵢ [ yⁱ·log(h_θ(xⁱ)) + (1 − yⁱ)·log(1 − h_θ(xⁱ)) ]
```
> Note: the handwritten notes show `1/2m`. The standard log-loss uses `1/m`. (The `1/2m` is a convention from MSE, where the 2 cancels when differentiating.) Either gives the same minimum, but `1/m` is standard.

This gives a **convex function**, so gradient descent reaches the global minimum.

### 1.8 Convergence algorithm (Gradient Descent)

```
Repeat until convergence {
    θⱼ := θⱼ − α · ∂J(θ₀, θ₁)/∂θⱼ          (for j = 0 and 1, update simultaneously)
}
```

Where `α` = learning rate. For logistic regression the derivative works out to a neat form:

```
∂J/∂θⱼ = (1/m) Σᵢ (h_θ(xⁱ) − yⁱ) · xⱼⁱ
```

(Same form as linear regression, but `h_θ` is the sigmoid.)

**Learning rate guide**

| α | Effect |
|---|---|
| Too small | Very slow convergence |
| Good | Cost steadily decreases |
| Too large | Overshoots, cost bounces or diverges |

**Tip:** plot J vs iterations. It should decrease smoothly.

### 1.9 Real-world uses

| Domain | Question | Output |
|---|---|---|
| Email | Is this spam? | Spam / Not spam |
| Banking | Will the customer default on a loan? | Default / No default |
| Healthcare | Does the patient have diabetes? | Yes / No |
| E-commerce | Will the user click the ad? | Click / No click |
| Telecom | Will the customer churn? | Churn / Stay |
| Fraud | Is this transaction fraudulent? | Fraud / Legit |
| HR | Will the employee leave? | Leave / Stay |

---

## 2. Performance Metrics

Topics: Confusion Matrix, Accuracy, Precision, Recall, F-Beta Score.

### 2.1 Confusion Matrix

A table comparing **actual** vs **predicted** values.

| | **Actual 1** | **Actual 0** |
|---|---|---|
| **Predicted 1** | **TP** (True Positive) | **FP** (False Positive) |
| **Predicted 0** | **FN** (False Negative) | **TN** (True Negative) |

- **TP**: predicted 1 and it really is 1 ✅
- **TN**: predicted 0 and it really is 0 ✅
- **FP** (Type I error): predicted 1 but actually 0 ❌
- **FN** (Type II error): predicted 0 but actually 1 ❌

**Example from the notes (7 predictions):**

| | Actual 1 | Actual 0 |
|---|---|---|
| Predicted 1 | TP = 3 | FP = 2 |
| Predicted 0 | FN = 1 | TN = 1 |

### 2.2 Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
(The notes have FN written twice by mistake; it should be FP + FN.)

Example: `(3 + 1) / 7 = 4/7 ≈ 57%`

### 2.3 The problem with accuracy: imbalanced datasets

Dataset of **1000 points**: 900 are class 1, 100 are class 0 (**imbalanced**).

If the model **always predicts 1**, accuracy is **90%**, yet the model is useless: it never detects a single class-0 case.

**Real example:** Fraud detection with 99.9% legit transactions. A model that says "never fraud" gets 99.9% accuracy but catches **zero** fraud. So we need Precision and Recall.

### 2.4 Precision

```
Precision = TP / (TP + FP)
```

**Meaning: of everything the model predicted as positive, how many were actually positive?**
"When the model says YES, how often is it right?"

Example: `3 / (3 + 2) = 0.60`

### 2.5 Recall (Sensitivity / True Positive Rate)

```
Recall = TP / (TP + FN)
```

**Meaning: of all the actual positives, how many did the model find?**
"Of all the real YES cases, how many did we catch?"

Example: `3 / (3 + 1) = 0.75`

> ⚠️ **Correction to the handwritten notes:** The notes describe Precision as "out of all actual values, how many are correctly predicted" and Recall as "out of all predicted values..." These are **swapped**. The formulas are right, but the descriptions above are the correct ones.
>
> **Memory trick:** Precision's denominator = *predicted* positives (TP + FP). Recall's denominator = *actual* positives (TP + FN).

### 2.6 When to use Precision vs Recall

**Use PRECISION when False Positives are costly.**

| Case | Why |
|---|---|
| **Spam filter** | Mail is *not spam* but the model says spam → the user misses an important email (a blunder). Mail is spam and the model says spam → good. So FP is the blunder → use **Precision**. |
| Recommending a video to kids | A wrong recommendation (FP) is harmful |
| Approving a loan | Approving a bad borrower (FP) loses money |
| Legal / criminal conviction | Convicting an innocent person (FP) is a serious error |

**Use RECALL when False Negatives are costly.**

| Case | Why |
|---|---|
| **Cancer / disease screening** | Missing a sick patient (FN) can cost a life. A false alarm just leads to another test |
| **Fraud detection** | Missing a fraud (FN) costs money |
| **Airport security** | Missing a threat (FN) is dangerous |
| **Machine failure prediction** | Missing a failure (FN) causes downtime |

**Trade-off:** Raising the threshold (e.g. 0.5 → 0.8) increases Precision but lowers Recall. Lowering it (0.5 → 0.2) increases Recall but lowers Precision.

### 2.7 F-Beta Score

When you need to balance both:

```
Fβ = (1 + β²) · (Precision · Recall) / (β² · Precision + Recall)
```
(The notes write the denominator as `Precision + Recall`, which is correct only when β = 1. The general form has `β² · Precision`.)

| Situation | β | Name | Formula |
|---|---|---|---|
| **FP and FN are both important** | 1 | **F1 score** (harmonic mean) | `2·P·R / (P + R)` |
| **FP is more important than FN** | 0.5 (β < 1) | F0.5 | `(1 + 0.25)·P·R / (0.25·P + R)` |
| **FN is more important than FP** | 2 (β > 1) | F2 | `(1 + 4)·P·R / (4·P + R)` |

**Why harmonic mean?** It punishes extreme values. If Precision = 1.0 and Recall = 0.0, the arithmetic mean is 0.5 (looks OK) but F1 = 0 (correctly shows the model is broken).

**Example (P = 0.60, R = 0.75):**
```
F1   = 2(0.6)(0.75) / (0.6 + 0.75)           = 0.667
F0.5 = 1.25(0.6)(0.75) / (0.25(0.6) + 0.75)  = 0.625   (leans toward precision)
F2   = 5(0.6)(0.75) / (4(0.6) + 0.75)        = 0.714   (leans toward recall)
```

### 2.8 Imbalanced example worked fully

1000 samples (900 class 1, 100 class 0). Model always predicts 1:

| | Actual 1 | Actual 0 |
|---|---|---|
| Predicted 1 | TP = 900 | FP = 100 |
| Predicted 0 | FN = 0 | TN = 0 |

- Accuracy = 900/1000 = **90%** (looks great)
- Precision (class 1) = 900/1000 = 0.90
- Recall (class 1) = 900/900 = 1.0
- But for class 0: Recall = 0/100 = **0** ← the model is useless for class 0

**Lesson:** always look at the confusion matrix and per-class metrics, not just accuracy.

### 2.9 Which metric should I use? (Quick guide)

| Situation | Metric |
|---|---|
| Balanced classes | Accuracy (OK) |
| Imbalanced classes | Precision, Recall, F1 |
| FP is costly | Precision / F0.5 |
| FN is costly | Recall / F2 |
| Balance of both | F1 |
| Compare models across thresholds | ROC-AUC / PR-AUC |

### 2.10 Bonus metrics worth knowing

- **Specificity (TNR)** = `TN / (TN + FP)`
- **ROC curve**: plots TPR (Recall) vs FPR (`FP / (FP + TN)`) across thresholds. **AUC** close to 1 means a good model, 0.5 means random.
- **PR curve**: Precision vs Recall; better than ROC for heavily imbalanced data.
- **Log loss**: the same cost function used in training (evaluates probability quality).

---

## 3. Multiclass Logistic Regression (One-versus-Rest)

### 3.1 Binary → Multiclass

Plain logistic regression handles **two** classes (Class A vs Class B), separated by one decision boundary.

For **3 or more classes** (e.g. Cat / Dog / Bird), we use **One-versus-Rest (OvR)**, also called One-vs-All.

### 3.2 Idea

Internally create **multiple models**, each acting as a **binary classifier**.

For 3 classes (1, 2, 3):

| Model | Positive class | "Rest" (negative) |
|---|---|---|
| **M1** | Class 1 | Classes 2 and 3 |
| **M2** | Class 2 | Classes 1 and 3 |
| **M3** | Class 3 | Classes 1 and 2 |

In the diagram, each coloured line (M1 yellow, M2 blue, M3 red) separates one class from the others.

### 3.3 Training data transformation

Original data (features f₁, f₂, f₃; output O₁/O₂/O₃):

| f₁ | f₂ | f₃ | Output |
|---|---|---|---|
| … | … | … | O₁ |
| … | … | … | O₂ |
| … | … | … | O₃ |

For **Model M1**, relabel: `O₁ → 1`, `O₂ → 0`, `O₃ → 0`.
For **Model M2**: `O₂ → 1`, others → 0. For **M3**: `O₃ → 1`, others → 0.

Each model trains as a normal binary logistic regression.

### 3.4 Prediction

Pass new data to **all** models and pick the **highest probability**:

```
New data →  M1 = 0.25
            M2 = 0.20
            M3 = 0.55   ✔ highest
```

**Predicted class = Class 3** (`argmax`).

### 3.5 Example

Classifying fruit as **Apple / Banana / Orange** from weight and colour:

- M1: "Apple vs not Apple" → 0.10
- M2: "Banana vs not Banana" → 0.05
- M3: "Orange vs not Orange" → 0.85 → **Orange**

### 3.6 Notes on OvR

- For **K classes**, you train **K models**.
- The probabilities from different models **don't necessarily sum to 1** (the notes' 0.25 + 0.20 + 0.55 happens to equal 1.0). You can normalise them if needed.
- Works well, simple, and interpretable. Struggles if classes overlap heavily.

### 3.7 Alternatives

| Method | Idea |
|---|---|
| **One-vs-One (OvO)** | One model per pair of classes: K(K−1)/2 models, then vote |
| **Softmax (Multinomial) regression** | One model, outputs a probability for each class that sums to 1. Default in scikit-learn for multiclass |

---

## 4. Full Python Example (scikit-learn)

### 4.1 Binary: Study hours → Pass/Fail

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             f1_score, fbeta_score, classification_report)

# Data
hours = np.array([1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]).reshape(-1, 1)
passed = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    hours, passed, test_size=0.33, random_state=42, stratify=passed)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
probs = model.predict_proba(X_test)[:, 1]   # P(y = 1)
preds = (probs >= 0.5).astype(int)          # custom threshold possible

# Evaluate
print("Confusion matrix:\n", confusion_matrix(y_test, preds))
print("Accuracy :", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds, zero_division=0))
print("Recall   :", recall_score(y_test, preds))
print("F1       :", f1_score(y_test, preds))
print("F2       :", fbeta_score(y_test, preds, beta=2))
print(classification_report(y_test, preds, zero_division=0))
```

### 4.2 Logistic regression from scratch (NumPy)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    b = 0.0
    for _ in range(epochs):
        z = X @ theta + b
        h = sigmoid(z)
        # gradients of log loss
        d_theta = (1 / m) * X.T @ (h - y)
        d_b = (1 / m) * np.sum(h - y)
        theta -= lr * d_theta
        b -= lr * d_b
    return theta, b

def log_loss(h, y):
    eps = 1e-9
    return -np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))
```

### 4.3 Multiclass One-vs-Rest

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

ovr = OneVsRestClassifier(LogisticRegression(max_iter=500))
ovr.fit(X_train, y_train)

print(len(ovr.estimators_))             # 3 models (one per class)
print(ovr.predict_proba(X_test[:1]))    # one score per class
print(ovr.predict(X_test[:1]))          # argmax class
print("Accuracy:", ovr.score(X_test, y_test))
```

### 4.4 Changing the threshold for imbalanced problems

```python
# Favor recall (e.g. disease screening): lower threshold
preds_high_recall = (probs >= 0.3).astype(int)

# Favor precision (e.g. spam filter): raise threshold
preds_high_precision = (probs >= 0.8).astype(int)

# Handle class imbalance directly
model = LogisticRegression(class_weight="balanced")
```

---

## 5. Assumptions, Pros and Cons

### Assumptions
- Target is binary (or handled via OvR/softmax for multiclass)
- Observations are independent
- **Linear relationship between features and the log-odds** (`log(p/(1−p)) = θ₀ + θ₁x`)
- Little to no multicollinearity between features
- Reasonably large sample size

### Pros
- Simple, fast, easy to implement
- Outputs **probabilities**, not just labels
- Interpretable coefficients (`e^θ` = odds ratio)
- Works well as a **baseline**
- Less prone to overfitting with regularization

### Cons
- Only a **linear decision boundary** (can't separate circles/XOR shapes without feature engineering)
- Sensitive to outliers (less than linear regression, but still affected) and multicollinearity
- Needs feature scaling for faster gradient descent convergence
- Struggles with complex non-linear data (use trees, SVM with kernels, neural nets)

### Improving the model
- **Regularization**: L1 (Lasso) or L2 (Ridge). In sklearn: `LogisticRegression(penalty="l2", C=1.0)`. Smaller `C` means stronger regularization.
- **Feature scaling**: StandardScaler
- **Polynomial features** to allow curved boundaries
- **Class weights / SMOTE** for imbalanced data

---

## 6. Quick Revision Cheat Sheet

| Concept | Formula / Rule |
|---|---|
| Hypothesis | `h(x) = 1 / (1 + e^(−z))`, `z = θ₀ + θ₁x` |
| Decision | `h ≥ 0.5` → 1 ; else 0 (i.e. `z ≥ 0`) |
| Why not MSE? | Non-convex with sigmoid |
| Cost (log loss) | `−(1/m) Σ [y log h + (1−y) log(1−h)]` |
| Gradient | `(1/m) Σ (h − y) xⱼ` |
| Update | `θⱼ := θⱼ − α ∂J/∂θⱼ` |
| Accuracy | `(TP+TN) / (TP+TN+FP+FN)` |
| Precision | `TP / (TP+FP)`, minimise **FP** |
| Recall | `TP / (TP+FN)`, minimise **FN** |
| F1 | `2PR / (P+R)` |
| Fβ | `(1+β²)PR / (β²P + R)` ; β<1 favors precision, β>1 favors recall |
| Multiclass OvR | K classes → K binary models → `argmax` of probabilities |

### Common interview questions
1. **Why is it called regression if it's classification?** It regresses on the log-odds (a continuous value), then maps to a probability with the sigmoid.
2. **Why log loss instead of MSE?** Convex with sigmoid, and it penalises confident wrong predictions strongly.
3. **Accuracy is 95% but the model is bad. Why?** Imbalanced data. Check Precision/Recall/F1.
4. **Spam filter: Precision or Recall?** Precision (FP = good mail lost).
5. **Cancer detection: Precision or Recall?** Recall (FN = missed patient).
6. **How does logistic regression handle multiple classes?** One-vs-Rest (K binary models) or Softmax.
7. **What if my data is not linearly separable?** Add polynomial features, or use SVM (kernel), trees, or neural nets.
