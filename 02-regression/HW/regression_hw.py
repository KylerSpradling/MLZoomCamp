import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv'

# read in the data
df = pd.read_csv(url)

# subset data to necessary columns
df = df[['engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg']]

# lookling at fuel_efficiency_mpg distribution
sns.histplot(df['fuel_efficiency_mpg'], bins=50, kde=True)
plt.title('Fuel Efficiency Distribution')
plt.ylabel('Car Count')
plt.xlabel('MPG')
plt.show()

# data appears to be normally distributed... no need to perform log transformation

# Checking for NULL Values
df.isnull().sum()
# Horsepower column has 708 NULL values

horsepower_median = df['horsepower'].median()
# horsepower median = 149

#Preparing the data

#1.) Determine the size of each subset
n = len(df)
n_val = int(n*.2)
n_test = int(n*.2)
n_train = n - (n_val + n_test)


#2.) Shuffling data
np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)
df_shuffle = df.iloc[idx]

#3.) Splitting the data
df_train = df_shuffle.iloc[:n_train].reset_index(drop=True)
df_val = df_shuffle.iloc[n_train:n_train + n_val].reset_index(drop=True)
df_test = df_shuffle.iloc[n_train + n_val:].reset_index(drop=True)

#4.) Storing the Target variable
y_train = df_train['fuel_efficiency_mpg']
y_val = df_val['fuel_efficiency_mpg']
y_test = df_test['fuel_efficiency_mpg']

base_feat = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']

def prepare_X(df: pd.DataFrame, NA_values: float = 0) -> np.ndarray:
    df_num = df[base_feat]
    df_num = df_num.fillna(NA_values)

    X = df_num.values

    return X

def train_linear_regression_simple(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    X0 = np.ones(X.shape[0])
    X = np.column_stack([X0,X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)

    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]

def rmse(y_actual: np.ndarray, y_pred: np.ndarray):
    error = y_actual - y_pred
    se = error**2
    mse = se.mean()
    return np.sqrt(mse)

horsepower_mean = df_train['horsepower'].mean()

# applying to train data
    # model using the mean on NA values
X_mean_train = prepare_X(df_train, NA_values=horsepower_mean)
w0_mean, w_mean = train_linear_regression_simple(X_mean_train, y_train)


X_mean_val = prepare_X(df_val, NA_values=horsepower_mean)
y_mean_pred = w0_mean + X_mean_val.dot(w_mean)

#Mean RMSE Score
RMSE_mean_score = np.round(rmse(y_val, y_mean_pred),2) #0.46

# Model using 0 on NULL values

# Train
X_zero_train = prepare_X(df_train, NA_values=0)
w0_zero,w_zero = train_linear_regression_simple(X_zero_train, y_train)

# Validation
X_zero_val = prepare_X(df_val, NA_values=0)
y_zero_pred = w0_zero + X_zero_val.dot(w_zero)

# RMSE Score
RMSE_zero_score = np.round(rmse(y_val, y_zero_pred),2) #0.52


# Regularized regression

def ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 0) -> tuple[float, np.ndarray]:
    X0 = np.ones(X.shape[0])
    X = np.column_stack([X0,X])

    XTX = X.T.dot(X)
    XTX = XTX + alpha * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)

    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]

alpha_score = [0, 0.01, 0.1, 1, 5, 10, 100]
rmse_scores = []

for a in alpha_score:
  w0, w = ridge_regression(X_zero_train, y_train, alpha=a)
  y_pred = w0 + X_zero_val.dot(w)
  scores = np.round(rmse(y_val, y_pred),2)

  rmse_scores.append((a,scores))

  print(f'alpha:{a} RMSE{scores}')

# all scores are the same, so we should keep a = 0


# using multiple seeds
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scores = []

for seed in seeds:
    #set random seed at start of loop
    np.random.seed(seed)

    n = len(df)
    n_val = int(n*.2)
    n_test = int(n*.2)
    n_train = n - (n_val + n_test)

    idx = np.arange(n)
    np.random.shuffle(idx)

    df_shuffle = df.iloc[idx]
    
    df_train = df_shuffle.iloc[:n_train].reset_index(drop=True)
    df_val = df_shuffle.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df_shuffle.iloc[n_train + n_val:].reset_index(drop=True)

    y_train = df_train['fuel_efficiency_mpg']
    y_val = df_val['fuel_efficiency_mpg']

    X_train = prepare_X(df_train, NA_values=0)
    w0, w = ridge_regression(X_train, y_train)

    X_val = prepare_X(df_val, NA_values=0)
    y_pred = w0 + X_val.dot(w)

    score = rmse(y_val, y_pred)

    scores.append(score)

    print(f'Seed {seed} -> RMSE {round(score,3)}')


np.round(np.std(scores),3)


### Question 6

    # Split the dataset like previously, use seed 9.
    # Combine train and validation datasets.
    # Fill the missing values with 0 and train a model with `r=0.001`. 
    # What's the RMSE on the test dataset?


np.random.seed(9)
idx = np.arange(n)
np.random.shuffle(idx)
df_shuffle = df.iloc[idx]

df_train = df_shuffle.iloc[:n_train].reset_index(drop=True)
df_val = df_shuffle.iloc[n_train:n_train + n_val].reset_index(drop=True)
df_test = df_shuffle.iloc[n_train + n_val:].reset_index(drop=True)

df_train_full = pd.concat([df_train,df_val])
X_train_full = prepare_X(df_train_full, NA_values=0)

y_train_full = np.concatenate([y_train,y_val])

w0,w = ridge_regression(X_train_full, y_train_full, alpha=0.001)

X_test = prepare_X(df_test,NA_values=0)

y_pred = w0 + X_test.dot(w)


np.round(rmse(y_test,y_pred),3)