# ML Zoomcamp - Quick Reference Summaries

## About This Document

This document serves as a quick-reference guide for the **ML Zoomcamp 2025** course, a practical machine learning bootcamp that takes you from Python fundamentals to building and evaluating predictive models. The course follows a logical progression: first establishing the mathematical foundation with NumPy, then applying supervised learning techniques (regression for continuous outcomes, classification for categorical outcomes), and finally learning how to properly evaluate model performance. Each section below contains key concepts, Python code examples, and essential formulas to help you quickly recall what you've learned.

---

## Table of Contents

- [01 - Introduction to ML & NumPy](#01---introduction-to-ml--numpy)
  - [1.7 NumPy Fundamentals](#17-numpy-fundamentals)
  - [1.8 Linear Algebra](#18-linear-algebra)
- [02 - Regression](#02---regression)
  - [Data Preparation & EDA](#22-23-data-preparation--eda)
  - [Validation Framework](#24-validation-framework)
  - [Linear Regression](#25-27-linear-regression)
  - [Model Evaluation (RMSE)](#28-210-model-evaluation-rmse)
  - [Feature Engineering](#211-212-feature-engineering--categorical-variables)
  - [Regularization](#213-214-regularization-ridge-regression)
- [03 - Classification (Churn Prediction)](#03---classification-churn-prediction)
  - [Data Preparation](#31-33-data-preparation)
  - [Feature Importance](#35-feature-importance-churn-rate--risk-ratio)
  - [One-Hot Encoding](#38-one-hot-encoding)
  - [Logistic Regression](#39-310-logistic-regression)
- [04 - Evaluation Metrics](#04---evaluation-metrics)
  - [Accuracy & Thresholds](#41-42-accuracy--threshold-selection)
  - [Confusion Matrix](#43-confusion-matrix)
  - [Precision & Recall](#44-precision--recall)
  - [ROC Curves & AUC](#45-roc-curves--auc)
- [Quick Reference Card](#quick-reference-card)
- [Common Imports Cheat Sheet](#common-imports-cheat-sheet)

---

## 01 - Introduction to ML & NumPy

NumPy is the backbone of numerical computing in Python and the foundation upon which most machine learning libraries are built. Before diving into ML algorithms, you need to understand how to efficiently manipulate arrays and perform linear algebra operations—these are the building blocks of every model you'll create.

### Key Concepts
- NumPy is the foundation for numerical computing in Python
- Arrays are fixed-size, unlike Python lists
- Linear algebra operations are essential for ML algorithms
- Pandas builds on NumPy for data manipulation

### Sub-topics

#### 1.7 NumPy Fundamentals

**Creating Arrays**
```python
import numpy as np

np.zeros(5)                    # [0., 0., 0., 0., 0.]
np.ones(10)                    # [1., 1., 1., 1., ...]
np.full(10, 2.5)               # [2.5, 2.5, 2.5, ...]
np.array([1, 2, 3, 5, 7])      # Convert list to array
np.arange(1, 54, 10)           # [1, 11, 21, 31, 41, 51] - step by 10
np.linspace(0, 1, 11)          #        [0, 0.1, 0.2, ..., 1.0] - 11 evenly spaced
```

**Multi-dimensional Arrays**
```python
# Create 2D array
n = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

n[1, 1]      # 5 - Access element at row 1, col 1
n[1]         # [4, 5, 6] - Access entire row
n[:, 1]      # [2, 5, 8] - Access entire column
n[0:2, 1]    # [2, 5] - Rows 0-1 from column 1
```

**Random Arrays**
```python
np.random.seed(42)                        # Set seed for reproducibility
np.random.rand(5, 2)                      # 5x2 uniform distribution [0,1)
np.random.randn(5, 2)                     # 5x2 standard normal distribution
np.random.randint(low=0, high=100, size=(5, 2))  # 5x2 random integers
```

**Element-wise Operations**
```python
a = np.arange(5)        # [0, 1, 2, 3, 4]
a + 1                   # [1, 2, 3, 4, 5]
a * 2                   # [0, 2, 4, 6, 8]
(10 + (a * 2)) ** 2 / 100  # Chained operations
```

**Comparison & Filtering**
```python
a = np.array([0, 1, 2, 3, 4])
a >= 2                  # [False, False, True, True, True]
a[a >= 2]               # [2, 3, 4] - Filter elements
```

**Summary Operations**
```python
a.min()    # 0
a.max()    # 4
a.mean()   # 2.0
a.sum()    # 10
a.std()    # 1.414...
```

#### 1.8 Linear Algebra

**Vector-Vector Multiplication (Dot Product)**
```python
u = np.array([2, 4, 5, 6])
v = np.array([1, 0, 0, 2])

u.dot(v)   # 14 (2*1 + 4*0 + 5*0 + 6*2)

# Manual implementation
def dot_product(u, v):
    result = 0
    for i in range(len(u)):
        result += u[i] * v[i]
    return result
```

**Matrix-Vector Multiplication**
```python
U = np.array([
    [2, 4, 5, 6],
    [1, 2, 1, 2],
    [3, 1, 2, 1]
])
v = np.array([1, 0, 0, 2])

U.dot(v)   # [14, 5, 5] - Each row dotted with v
```

**Matrix-Matrix Multiplication**
```python
X = np.array([[1, 2], [3, 4], [5, 6]])

XTX = X.T.dot(X)           # X-transpose times X
XTX_inv = np.linalg.inv(XTX)  # Inverse of XTX
```

### Key Formulas
```
Dot Product: u · v = Σ(u[i] × v[i])
Matrix Inverse: XTX⁻¹ where XTX = X^T × X
```

---

## 02 - Regression

Regression is your first supervised learning algorithm, used when predicting continuous numerical values like prices, temperatures, or scores. The core workflow involves splitting your data to prevent overfitting, training a model to find the relationship between features and target, and evaluating performance using metrics like RMSE. This module also introduces regularization—a technique to prevent your model from memorizing the training data.

### Key Concepts
- Regression predicts continuous values (e.g., car prices)
- Train/Validation/Test split prevents overfitting
- Feature engineering improves model performance
- Regularization prevents overfitting by penalizing large weights

### Sub-topics

#### 2.2-2.3 Data Preparation & EDA
```python
import pandas as pd

df = pd.read_csv('data.csv')
df.head()                    # View first 5 rows
df.isnull().sum()            # Count missing values per column
df.describe()                # Summary statistics
df['price'].hist()           # Distribution of target
```

#### 2.4 Validation Framework
```python
from sklearn.model_selection import train_test_split

# Split: 60% train, 20% val, 20% test
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# Extract target variable
y_train = df_train['price'].values
y_val = df_val['price'].values
y_test = df_test['price'].values

# Remove target from features
del df_train['price']
del df_val['price']
del df_test['price']
```

#### 2.5-2.7 Linear Regression

**Training with Normal Equation**
```python
def train_linear_regression(X, y):
    # Add bias column (column of ones)
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    # Normal equation: w = (X^T X)^(-1) X^T y
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]  # bias, weights

w0, w = train_linear_regression(X_train, y_train)
```

**Making Predictions**
```python
def predict(X, w0, w):
    return w0 + X.dot(w)

y_pred = predict(X_val, w0, w)
```

#### 2.8-2.10 Model Evaluation (RMSE)
```python
def rmse(y_true, y_pred):
    error = y_pred - y_true
    mse = (error ** 2).mean()
    return np.sqrt(mse)

score = rmse(y_val, y_pred)
print(f'RMSE: {score:.4f}')
```

#### 2.11-2.12 Feature Engineering & Categorical Variables
```python
# Simple feature engineering
df['age'] = 2024 - df['year']

# One-hot encoding with DictVectorizer
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
train_dicts = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

# For validation (don't fit, just transform)
val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)
```

#### 2.13-2.14 Regularization (Ridge Regression)
```python
def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    # Add regularization term to diagonal
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]

# Tune regularization parameter
for r in [0, 0.001, 0.01, 0.1, 1, 10]:
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = predict(X_val, w0, w)
    print(f'r={r}: RMSE={rmse(y_val, y_pred):.4f}')
```

### Key Formulas
```
Linear Regression:  g(x) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
Vector Form:        g(x) = w₀ + w^T × X
Normal Equation:    w = (X^T X)^(-1) X^T y
RMSE:               √(1/n × Σ(y_pred - y_actual)²)
Ridge Regression:   w = (X^T X + λI)^(-1) X^T y
```

---

## 03 - Classification (Churn Prediction)

Classification differs from regression in that you're predicting categories rather than numbers. In binary classification, you predict one of two outcomes (yes/no, churn/stay, spam/not spam). The key insight is that logistic regression outputs a probability, which you then convert to a prediction using a threshold. This module also covers feature importance techniques to understand which variables actually influence your predictions.

### Key Concepts
- Binary classification predicts 0 or 1 (e.g., churn/no churn)
- Feature importance helps identify which features matter most
- Logistic regression outputs probabilities between 0 and 1
- Threshold determines the decision boundary (default: 0.5)

### Sub-topics

#### 3.1-3.3 Data Preparation
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('churn.csv')

# Standardize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Standardize categorical values
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Convert target to binary
df['churn'] = (df['churn'] == 'yes').astype(int)

# Train/val/test split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
```

#### 3.4 Exploratory Data Analysis
```python
# Check for missing values
df.isnull().sum()

# Target distribution
df['churn'].value_counts(normalize=True)

# Identify feature types
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'contract', 'paymentmethod']
```

#### 3.5 Feature Importance: Churn Rate & Risk Ratio
```python
# Global churn rate
global_churn = df_full_train['churn'].mean()  # e.g., 0.27

# Churn rate by category
churn_by_contract = df_full_train.groupby('contract')['churn'].mean()

# Risk ratio
risk_ratio = churn_by_contract / global_churn
# month-to-month: 1.6 (higher risk)
# two_year: 0.1 (lower risk)

# Difference from global
diff = global_churn - churn_by_contract
# Positive = churns less, Negative = churns more
```

#### 3.6 Mutual Information
```python
from sklearn.metrics import mutual_info_score

# Calculate for each categorical feature
def mutual_info_churn_score(series):
    return mutual_info_score(df_full_train['churn'], series)

mi_scores = df_full_train[categorical].apply(mutual_info_churn_score)
mi_scores.sort_values(ascending=False)
# contract: 0.098 (most informative)
# gender: 0.0001 (least informative)
```

#### 3.7 Correlation
```python
# Correlation with target
df_full_train[numerical].corrwith(df_full_train['churn'])
# tenure: -0.35 (longer tenure = less churn)
# monthlycharges: 0.20 (higher charges = more churn)
```

#### 3.8 One-Hot Encoding
```python
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)  # Don't fit on validation!

# See feature names
dv.get_feature_names_out()
# ['contract=month-to-month', 'contract=one_year', ...]
```

#### 3.9-3.10 Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get probabilities (soft predictions)
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Get hard predictions with threshold
threshold = 0.5
y_pred = (y_pred_proba >= threshold).astype(int)

# Accuracy
accuracy = (y_pred == y_val).mean()
```

#### 3.11-3.12 Model Interpretation
```python
# View coefficients
feature_names = dv.get_feature_names_out()
coef_dict = dict(zip(feature_names, model.coef_[0]))

# Positive coef = increases churn probability
# contract=month-to-month: 0.97
# Negative coef = decreases churn probability
# contract=two_year: -0.95
```

### Key Formulas
```
Churn Rate:    (Churned in Group) / (Total in Group)
Risk Ratio:    (Group Churn Rate) / (Global Churn Rate)
Sigmoid:       σ(z) = 1 / (1 + e^(-z))
Logistic Reg:  P(y=1|x) = σ(w₀ + w^T × x)
```

---

## 04 - Evaluation Metrics

Knowing how to build a model is only half the battle—you also need to know if it's actually good. Accuracy seems intuitive but can be misleading, especially when your classes are imbalanced (e.g., 95% of customers don't churn). This module teaches you to look beyond accuracy using the confusion matrix, precision, recall, and ROC curves. Understanding these metrics helps you choose the right model for your specific business problem.

### Key Concepts
- Accuracy alone can be misleading (especially with imbalanced classes)
- Confusion matrix shows all prediction outcomes
- Precision vs Recall is a trade-off
- ROC-AUC provides threshold-independent evaluation

### Sub-topics

#### 4.1-4.2 Accuracy & Threshold Selection
```python
from sklearn.metrics import accuracy_score

# Basic accuracy
accuracy = accuracy_score(y_val, y_pred)

# Test different thresholds
thresholds = np.linspace(0, 1, 21)
for t in thresholds:
    y_pred = (y_pred_proba >= t).astype(int)
    acc = accuracy_score(y_val, y_pred)
    print(f'Threshold: {t:.2f}, Accuracy: {acc:.3f}')
```

#### 4.3 Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calculate confusion matrix
actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
predict_positive = (y_pred == 1)
predict_negative = (y_pred == 0)

tp = (predict_positive & actual_positive).sum()  # True Positive
tn = (predict_negative & actual_negative).sum()  # True Negative
fp = (predict_positive & actual_negative).sum()  # False Positive
fn = (predict_negative & actual_positive).sum()  # False Negative

# Using sklearn
cm = confusion_matrix(y_val, y_pred)
# [[TN, FP],
#  [FN, TP]]

# Visualize
ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
```

#### 4.4 Precision & Recall
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Manual calculation
precision = tp / (tp + fp)  # When we predict positive, how often are we right?
recall = tp / (tp + fn)     # Of all positives, how many did we catch?
f1 = 2 * (precision * recall) / (precision + recall)

# Using sklearn
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Trade-off: Lower threshold = higher recall, lower precision
```

#### 4.5 ROC Curves & AUC
```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

# Plot ROC curve
plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# AUC score (0.5 = random, 1.0 = perfect)
auc = roc_auc_score(y_val, y_pred_proba)
print(f'AUC: {auc:.3f}')
```

### Key Formulas
```
Accuracy:    (TP + TN) / (TP + TN + FP + FN)
Precision:   TP / (TP + FP)
Recall:      TP / (TP + FN)
F1 Score:    2 × (Precision × Recall) / (Precision + Recall)
TPR:         TP / (TP + FN)  [same as Recall]
FPR:         FP / (FP + TN)
```

---

## Quick Reference Card

| Metric | Formula | When to Use |
|--------|---------|-------------|
| RMSE | `np.sqrt(((y_pred - y_true)**2).mean())` | Regression |
| Accuracy | `(y_pred == y_true).mean()` | Balanced classes |
| Precision | `tp / (tp + fp)` | Costly false positives |
| Recall | `tp / (tp + fn)` | Costly false negatives |
| F1 | `2 * p * r / (p + r)` | Balance P and R |
| AUC | `roc_auc_score(y_true, y_proba)` | Model comparison |

---

## Common Imports Cheat Sheet

```python
# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, mutual_info_score
)
```

---

*Generated from ML Zoomcamp 2025 notebooks*
