import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    sns.set()

    # Load data
    train_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
    test_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test.csv')
    
    print(f"Loading data from: {train_path} and {test_path}")
    
    if not os.path.exists(train_path):
        print(f"Error: Train file not found at {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"Error: Test file not found at {test_path}")
        return

    train = pd.read_csv(train_path, index_col='Id')
    test = pd.read_csv(test_path, index_col='Id')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # Data Exploration & Visualization (Commented out for automation)
    # num_data = train.select_dtypes(exclude=['object']).drop('SalePrice', axis=1).copy()
    # num_features = num_data.columns
    # missing_values = train.isnull().sum()
    # missing_values = missing_values[missing_values > 0]
    # missing_values.sort_values(inplace=True)
    # missing_values.plot.bar()
    # plt.show()

    target = train['SalePrice']

    # Dropping outliers
    train = train.drop(train[train['LotFrontage'] > 200].index)
    train = train.drop(train[train['LotArea'] > 100000].index)
    train = train.drop(train[train['MasVnrArea'] > 1500].index)
    train = train.drop(train[train['BsmtFinSF1'] > 4000].index)
    train = train.drop(train[train['TotalBsmtSF'] > 4000].index)
    train = train.drop(train[train['1stFlrSF'] > 4000].index)
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 400000)].index)
    train = train.drop(train[train['EnclosedPorch'] > 500].index)
    train = train.drop(train[(train['LowQualFinSF'] > 500) & (train['SalePrice'] > 400000)].index)

    # Dropping correlated columns
    train = train.drop(['GarageYrBlt', '1stFlrSF', 'TotRmsAbvGrd', 'GarageArea'], axis=1)
    test = test.drop(['GarageYrBlt', '1stFlrSF', 'TotRmsAbvGrd', 'GarageArea'], axis=1)

    # Dropping columns with high missing values
    train = train.drop(['PoolQC', 'MiscFeature', 'Alley', 'PoolArea'], axis=1)
    test = test.drop(['PoolQC', 'MiscFeature', 'Alley', 'PoolArea'], axis=1)

    # FEATURE ENGINEERING
    # Fill LotFrontage
    train['LotFrontage'] = train.LotFrontage.fillna(np.sqrt(train.LotArea))
    test['LotFrontage'] = test.LotFrontage.fillna(np.sqrt(test.LotArea))

    # Fill MasVnrArea
    train['MasVnrArea'] = train.MasVnrArea.fillna(0.5 * (test.MasVnrArea.mean() + train.MasVnrArea.mean()))
    test['MasVnrArea'] = test.MasVnrArea.fillna(0.5 * (test.MasVnrArea.mean() + train.MasVnrArea.mean()))

    # Fill categorical missing values
    cat_missing_values = ['Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual',
                          'GarageType', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond', 'MasVnrType']

    for column in cat_missing_values:
        train[column] = train[column].fillna("None")
    for column in cat_missing_values:
        test[column] = test[column].fillna("None")

    # Target Logic
    y = np.log(train.SalePrice)
    X = train.drop(['SalePrice'], axis=1)

    # One-hot-encoding
    print("One-hot encoding data...")
    X = pd.get_dummies(X)

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Imputation
    my_imputer = SimpleImputer()
    train_X = my_imputer.fit_transform(train_X)
    val_X = my_imputer.transform(val_X)

    # Helper to inverse transform target
    def inv_y(transformed_y):
        return np.exp(transformed_y)

    # Linear Regression
    print("Training Linear Regression...")
    linear_model = LinearRegression()
    linear_model.fit(train_X, train_y)
    linear_val_predictions = linear_model.predict(val_X)
    linear_val_mae = mean_absolute_error(inv_y(linear_val_predictions), inv_y(val_y))
    print("Validation MAE for Linear Regression Model: {:,.0f}".format(linear_val_mae))

    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(random_state=5)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(inv_y(rf_val_predictions), inv_y(val_y))
    print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

    # Test Data Processing
    print("Processing Test Data...")
    test['BsmtFullBath'] = test['BsmtFullBath'].fillna(0)
    test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(0)
    test['GarageCars'] = test['GarageCars'].fillna(0)
    test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(0)
    test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(0)
    test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(0)
    test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)

    # One-hot encoding for test data
    test_X = pd.get_dummies(test)

    # Align columns
    final_train, final_test = X.align(test_X, join='left', axis=1)

    # Impute test data
    final_test_imputed = my_imputer.transform(final_test)

    # Predictions
    print("Generating predictions...")
    test_preds = rf_model.predict(final_test_imputed)

    # Output
    output = pd.DataFrame({'Id': test.index, 'SalePrice': inv_y(test_preds)})
    output_path = 'submission.csv'
    output.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    main()
