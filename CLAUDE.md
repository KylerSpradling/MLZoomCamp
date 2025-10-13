# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About MLZoomCamp

This directory contains coursework for the Machine Learning Zoomcamp - a structured course covering fundamental machine learning concepts with hands-on implementations. The focus is on understanding algorithms by implementing them from scratch using NumPy, alongside using scikit-learn.

## Directory Structure

- `01-intro/` - Introduction to NumPy and basic data manipulation
- `02-regression/` - Linear and ridge regression implementations
- `03-classification/` - Logistic regression and classification tasks
- Each module contains:
  - `HW/` - Homework assignments with questions and solutions
  - `Notebook/` - Course notes and exploratory analysis

## Homework Structure

Each homework folder follows this pattern:
- `homework.md` or `homework_questions.md` - Assignment requirements
- `.ipynb` files - Interactive Jupyter notebooks with solutions
- `.py` files - Clean Python scripts (when provided)

## Core Implementation Conventions

### Data Preparation

**Missing Value Handling:**
```python
# Categorical features: Replace with 'NA' string
for col in cat_cols:
    df[col] = df[col].fillna('NA')

# Numerical features: Replace with 0.0 or mean
df['numerical_col'] = df['numerical_col'].fillna(0)
# OR
mean_value = df_train['numerical_col'].mean()  # Only use training data!
df['numerical_col'] = df['numerical_col'].fillna(mean_value)
```

**Data Splitting (Manual NumPy approach):**
```python
# Standard 60%/20%/20% split with random_state=42
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - (n_val + n_test)

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)
df_shuffle = df.iloc[idx]

df_train = df_shuffle.iloc[:n_train].reset_index(drop=True)
df_val = df_shuffle.iloc[n_train:n_train + n_val].reset_index(drop=True)
df_test = df_shuffle.iloc[n_train + n_val:].reset_index(drop=True)

# Always reset index and separate target variable
y_train = df_train['target']
y_val = df_val['target']
y_test = df_test['target']

del df_train['target']
del df_val['target']
del df_test['target']
```

### Regression Implementation

**Linear Regression (from scratch):**
```python
def train_linear_regression_simple(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    # Add bias term (column of ones)
    X0 = np.ones(X.shape[0])
    X = np.column_stack([X0, X])

    # Normal equation: w = (X^T X)^-1 X^T y
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]  # bias, weights
```

**Ridge Regression (with regularization):**
```python
def ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 0) -> tuple[float, np.ndarray]:
    X0 = np.ones(X.shape[0])
    X = np.column_stack([X0, X])

    XTX = X.T.dot(X)
    XTX = XTX + alpha * np.eye(XTX.shape[0])  # Add regularization

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]
```

**RMSE Calculation:**
```python
def rmse(y_actual: np.ndarray, y_pred: np.ndarray):
    error = y_actual - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)
```

**Feature Preparation Pattern:**
```python
def prepare_X(df: pd.DataFrame, NA_values: float = 0) -> np.ndarray:
    df_num = df[feature_columns]
    df_num = df_num.fillna(NA_values)
    X = df_num.values
    return X
```

### Classification Implementation

**Logistic Regression (sklearn with standard parameters):**
```python
from sklearn.linear_model import LogisticRegression

# Standard parameters for reproducibility
model = LogisticRegression(
    solver='liblinear',
    C=1.0,
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)
```

**One-Hot Encoding for Categorical Features:**
```python
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)

# Transform training data
dict_train = df_train.to_dict(orient='records')
X_train = dv.fit_transform(dict_train)

# Transform validation/test data
dict_val = df_val.to_dict(orient='records')
X_val = dv.transform(dict_val)  # Use transform, not fit_transform!
```

**Accuracy Calculation:**
```python
y_pred = model.predict(X_val)
accuracy = (y_pred == y_val).mean()
accuracy_rounded = accuracy.round(2)  # Round as specified in question
```

**Mutual Information Score:**
```python
from sklearn.metrics import mutual_info_score

# Calculate for categorical features vs target
score = mutual_info_score(y_train, df_train['categorical_feature'])
score_rounded = round(score, 2)
```

## Common Patterns and Best Practices

### Variable Naming Conventions
- `df_train`, `df_val`, `df_test` - Training, validation, and test dataframes
- `y_train`, `y_val`, `y_test` - Target variables
- `X_train`, `X_val`, `X_test` - Feature matrices
- `w0`, `w` - Bias term and weights (separated)
- Descriptive prefixes for variations: `X_mean_train`, `X_zero_val`, etc.

### Homework Workflow

1. **Read the assignment** - Check `homework.md`/`homework_questions.md` for exact requirements
2. **Load data** - Datasets are typically from `https://raw.githubusercontent.com/alexeygrigorev/datasets/master/`
3. **Handle missing values** - Follow the categorical ('NA') vs numerical (0 or mean) convention
4. **Split data** - Use 60%/20%/20% with `random_state=42` unless specified otherwise
5. **Separate target** - Store target in separate variables, remove from feature dataframes
6. **Train models** - Use exact parameters specified in questions
7. **Evaluate** - Round results as specified in each question
8. **Follow instructions precisely** - Questions specify exact parameters, rounding, and comparison methods

### Common Pitfalls to Avoid

1. **Don't access deleted target columns**
   ```python
   # WRONG - target was deleted from df_train
   mutual_info_score(df_train['converted'], df_train['feature'])

   # CORRECT - use separate y_train variable
   mutual_info_score(y_train, df_train['feature'])
   ```

2. **Don't round intermediate calculations**
   ```python
   # WRONG - affects subsequent calculations
   base_accuracy = (y_pred == y_val).mean().round(2)
   diff = base_accuracy - other_accuracy  # Precision lost!

   # CORRECT - only round final output
   base_accuracy = (y_pred == y_val).mean()
   diff = base_accuracy - other_accuracy
   print(round(diff, 2))  # Round only for display
   ```

3. **Use training data only for statistics**
   ```python
   # CORRECT - compute mean from training set only
   mean_value = df_train['feature'].mean()
   df_train['feature'] = df_train['feature'].fillna(mean_value)
   df_val['feature'] = df_val['feature'].fillna(mean_value)
   df_test['feature'] = df_test['feature'].fillna(mean_value)
   ```

4. **Don't fit DictVectorizer on validation/test data**
   ```python
   # WRONG
   X_val = dv.fit_transform(dict_val)

   # CORRECT
   X_val = dv.transform(dict_val)
   ```

5. **Parameter naming: C vs alpha**
   - In sklearn's LogisticRegression: `C` is the regularization parameter (higher C = less regularization)
   - In ridge regression: `alpha` or `r` is the regularization parameter (higher alpha = more regularization)
   - Don't confuse them in print statements!

## Key Learning Objectives

This coursework emphasizes:
- **Understanding over black-box usage** - Implementing algorithms manually before using libraries
- **Reproducibility** - Consistent use of random seeds and exact parameters
- **Educational rigor** - Manual RMSE calculations, custom regression functions
- **Real-world ML workflow** - Proper train/val/test splits, avoiding data leakage
