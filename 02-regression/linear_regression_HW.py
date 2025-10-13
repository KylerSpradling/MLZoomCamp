import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv'
df = pd.read_csv(url)

df = df[['engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg']]

# look at distributions of features
sns.histplot(df.fuel_efficiency_mpg, bins=20, kde=True)
plt.xlabel('Fuel Efficiency (mpg)')
plt.ylabel('Car Count')
plt.title('Distribution of Fuel Efficiency')
plt.show()
# data appears to be normally distributed, no need for log transformation

#look for missing values
df.isnull().sum()
# horsepower is missing some values


# Median Horsepower
df['horsepower'].mean()

# Prepare the data for modeling
n = len(df)
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

# Shuffle the data
np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)
df = df.iloc[idx]

# Split the data
df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train+n_val]
df_test = df.iloc[n_train+n_val:]

# values from the target column
y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values

# drop target column from the feature set
del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']


# preprocess the data
def prepare_X(df):
    base_features = ['engine_displacement','horsepower','vehicle_weight','model_year']
    df = df[base_features]
    df = df.copy()
    # fill missing values with the mean
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())
    # convert to numpy array
    X = df.values
    return X

def linear_regression(X, y):
    # add bias term
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    # calculate weights using the normal equation
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w[0], w[1:]

def predict(X, w0, w):
    # add bias term
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    y_pred = X.dot(np.concatenate([[w0], w]))
    return y_pred

# train the model
X_train = prepare_X(df_train)
w0, w = linear_regression(X_train, y_train)
print('Intercept:', w0)
print('Coefficients:', w)

# make predictions
X_val = prepare_X(df_val)
y_pred = predict(X_val, w0, w)

# evaluate the model
def rmse(y, y_pred):
    return np.sqrt(((y - y_pred) ** 2).mean())

print('RMSE:', rmse(y_val, y_pred))


