import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================================
# DATA LOADING AND EXPLORATION
# ============================================================================

def load_data(url: str) -> pd.DataFrame:
    """Load and prepare the fuel efficiency dataset."""
    df = pd.read_csv(url)
    df = df[['engine_displacement', 'horsepower', 'vehicle_weight', 
             'model_year', 'fuel_efficiency_mpg']]
    return df


def explore_data(df: pd.DataFrame):
    """Visualize fuel efficiency distribution and check for missing values."""
    # Plot distribution
    sns.histplot(df['fuel_efficiency_mpg'], bins=50, kde=True)
    plt.title('Fuel Efficiency Distribution')
    plt.ylabel('Car Count')
    plt.xlabel('MPG')
    plt.show()
    
    # Check for missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Calculate median for horsepower
    horsepower_median = df['horsepower'].median()
    print(f"\nHorsepower median: {horsepower_median}")


# ============================================================================
# DATA PREPARATION
# ============================================================================

def split_data(df: pd.DataFrame, seed: int = 42, 
               val_size: float = 0.2, test_size: float = 0.2) -> tuple:
    """Split data into train, validation, and test sets."""
    n = len(df)
    n_val = int(n * val_size)
    n_test = int(n * test_size)
    n_train = n - (n_val + n_test)
    
    # Shuffle data
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    df_shuffle = df.iloc[idx]
    
    # Split
    df_train = df_shuffle.iloc[:n_train].reset_index(drop=True)
    df_val = df_shuffle.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df_shuffle.iloc[n_train + n_val:].reset_index(drop=True)
    
    # Extract target variables
    y_train = df_train['fuel_efficiency_mpg'].values
    y_val = df_val['fuel_efficiency_mpg'].values
    y_test = df_test['fuel_efficiency_mpg'].values
    
    return df_train, df_val, df_test, y_train, y_val, y_test


def prepare_X(df: pd.DataFrame, NA_values: float = 0, 
              base_features: list = None) -> np.ndarray:
    """Prepare feature matrix by filling missing values."""
    if base_features is None:
        base_features = ['engine_displacement', 'horsepower', 
                        'vehicle_weight', 'model_year']
    
    df_num = df[base_features]
    df_num = df_num.fillna(NA_values)
    X = df_num.values
    
    return X


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_linear_regression_simple(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train linear regression model using normal equation."""
    X0 = np.ones(X.shape[0])
    X = np.column_stack([X0, X])
    
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]


def ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 0) -> tuple:
    """Train ridge regression model with regularization."""
    X0 = np.ones(X.shape[0])
    X = np.column_stack([X0, X])
    
    XTX = X.T.dot(X)
    XTX = XTX + alpha * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def rmse(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate root mean squared error."""
    error = y_actual - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)


def compare_imputation_strategies(df_train: pd.DataFrame, df_val: pd.DataFrame,
                                  y_train: np.ndarray, y_val: np.ndarray) -> dict:
    """Compare mean vs zero imputation for missing values."""
    horsepower_mean = df_train['horsepower'].mean()
    
    # Mean imputation
    X_mean_train = prepare_X(df_train, NA_values=horsepower_mean)
    w0_mean, w_mean = train_linear_regression_simple(X_mean_train, y_train)
    X_mean_val = prepare_X(df_val, NA_values=horsepower_mean)
    y_mean_pred = w0_mean + X_mean_val.dot(w_mean)
    rmse_mean = np.round(rmse(y_val, y_mean_pred), 2)
    
    # Zero imputation
    X_zero_train = prepare_X(df_train, NA_values=0)
    w0_zero, w_zero = train_linear_regression_simple(X_zero_train, y_train)
    X_zero_val = prepare_X(df_val, NA_values=0)
    y_zero_pred = w0_zero + X_zero_val.dot(w_zero)
    rmse_zero = np.round(rmse(y_val, y_zero_pred), 2)
    
    print(f"Mean Imputation RMSE: {rmse_mean}")
    print(f"Zero Imputation RMSE: {rmse_zero}")
    
    return {
        'mean_imputation': rmse_mean,
        'zero_imputation': rmse_zero
    }


def evaluate_regularization(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           alpha_values: list = None) -> list:
    """Evaluate different regularization parameters."""
    if alpha_values is None:
        alpha_values = [0, 0.01, 0.1, 1, 5, 10, 100]
    
    rmse_scores = []
    
    for alpha in alpha_values:
        w0, w = ridge_regression(X_train, y_train, alpha=alpha)
        y_pred = w0 + X_val.dot(w)
        score = np.round(rmse(y_val, y_pred), 2)
        rmse_scores.append((alpha, score))
        print(f'alpha: {alpha} RMSE: {score}')
    
    return rmse_scores


def evaluate_multiple_seeds(df: pd.DataFrame, seeds: list = None) -> tuple:
    """Evaluate model stability across different random seeds."""
    if seeds is None:
        seeds = list(range(10))
    
    scores = []
    
    for seed in seeds:
        df_train, df_val, df_test, y_train, y_val, y_test = split_data(df, seed=seed)
        
        X_train = prepare_X(df_train, NA_values=0)
        w0, w = ridge_regression(X_train, y_train)
        
        X_val = prepare_X(df_val, NA_values=0)
        y_pred = w0 + X_val.dot(w)
        
        score = rmse(y_val, y_pred)
        scores.append(score)
        print(f'Seed {seed} -> RMSE {round(score, 3)}')
    
    std_dev = np.round(np.std(scores), 3)
    print(f"\nStandard deviation of RMSE: {std_dev}")
    
    return scores, std_dev


# ============================================================================
# FINAL MODEL TRAINING
# ============================================================================

def train_final_model(df: pd.DataFrame, seed: int = 9, 
                     alpha: float = 0.001) -> tuple:
    """Train final model on combined train+val set and evaluate on test set."""
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df, seed=seed)
    
    # Combine train and validation
    df_train_full = pd.concat([df_train, df_val])
    X_train_full = prepare_X(df_train_full, NA_values=0)
    y_train_full = np.concatenate([y_train, y_val])
    
    # Train model
    w0, w = ridge_regression(X_train_full, y_train_full, alpha=alpha)
    
    # Evaluate on test set
    X_test = prepare_X(df_test, NA_values=0)
    y_pred = w0 + X_test.dot(w)
    
    test_rmse = np.round(rmse(y_test, y_pred), 3)
    print(f"Final Test RMSE: {test_rmse}")
    
    return w0, w, test_rmse


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv'
    df = load_data(url)
    
    # Explore data
    explore_data(df)
    
    # Split data
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df, seed=42)
    
    # Compare imputation strategies
    print("\n" + "="*50)
    print("Comparing Imputation Strategies")
    print("="*50)
    imputation_results = compare_imputation_strategies(df_train, df_val, y_train, y_val)
    
    # Evaluate regularization
    print("\n" + "="*50)
    print("Evaluating Regularization Parameters")
    print("="*50)
    X_zero_train = prepare_X(df_train, NA_values=0)
    X_zero_val = prepare_X(df_val, NA_values=0)
    reg_results = evaluate_regularization(X_zero_train, y_train, X_zero_val, y_val)
    
    # Evaluate stability across seeds
    print("\n" + "="*50)
    print("Evaluating Model Stability Across Seeds")
    print("="*50)
    scores, std = evaluate_multiple_seeds(df)
    
    # Train final model
    print("\n" + "="*50)
    print("Training Final Model")
    print("="*50)
    w0_final, w_final, final_rmse = train_final_model(df, seed=9, alpha=0.001)