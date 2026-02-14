import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# --- Custom Transformers to replicate main.py logic ---

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Only drop columns that exist in X
        cols = [c for c in self.cols_to_drop if c in X.columns]
        return X.drop(columns=cols)

class LotFrontageImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # train['LotFrontage'] = train.LotFrontage.fillna(np.sqrt(train.LotArea))
        if 'LotFrontage' in X.columns and 'LotArea' in X.columns:
            X['LotFrontage'] = X['LotFrontage'].fillna(np.sqrt(X['LotArea']))
        return X

class MasVnrAreaImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Calculate mean from training (simulated logic from main.py)
        # main.py used 0.5 * (test_mean + train_mean), which is weird but we'll specific fixed value or just mean
        # For simplicity and robustness in pipeline, we'll use simple mean of the seen data
        if 'MasVnrArea' in X.columns:
            self.fill_value_ = X['MasVnrArea'].mean()
        else:
            self.fill_value_ = 0
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'MasVnrArea' in X.columns:
            X['MasVnrArea'] = X['MasVnrArea'].fillna(self.fill_value_)
        return X

class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[col] = X[col].fillna("None")
        return X

class ZeroImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        return X

class NumericImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=['number']).columns
        self.imputer = SimpleImputer(strategy='median')
        self.imputer.fit(X[self.numeric_cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numeric_cols] = self.imputer.transform(X[self.numeric_cols])
        return X
        
class PandasGetDummies(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Record columns after dummying to ensure alignment during transform
        self.train_columns_ = pd.get_dummies(X).columns
        return self

    def transform(self, X):
        X_d = pd.get_dummies(X)
        # Align with training columns (add missing as 0, drop extras)
        # This mimics the "align" step in main.py
        X_d = X_d.reindex(columns=self.train_columns_, fill_value=0)
        return X_d

# --- Main Training Script ---

def train():
    print("Loading data...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col='Id')
    
    # 1. Outlier Removal (Rows only - done before pipeline)
    print("Removing outliers...")
    train_df = train_df.drop(train_df[train_df['LotFrontage'] > 200].index, errors='ignore')
    train_df = train_df.drop(train_df[train_df['LotArea'] > 100000].index, errors='ignore')
    train_df = train_df.drop(train_df[train_df['MasVnrArea'] > 1500].index, errors='ignore')
    train_df = train_df.drop(train_df[train_df['BsmtFinSF1'] > 4000].index, errors='ignore')
    train_df = train_df.drop(train_df[train_df['TotalBsmtSF'] > 4000].index, errors='ignore')
    train_df = train_df.drop(train_df[train_df['1stFlrSF'] > 4000].index, errors='ignore')
    train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 400000)].index, errors='ignore')
    train_df = train_df.drop(train_df[train_df['EnclosedPorch'] > 500].index, errors='ignore')
    train_df = train_df.drop(train_df[(train_df['LowQualFinSF'] > 500) & (train_df['SalePrice'] > 400000)].index, errors='ignore')

    # Target Transform
    y = np.log(train_df['SalePrice'])
    X = train_df.drop(['SalePrice'], axis=1)

    # 2. Define Columns for Transformers
    cols_to_drop = [
        'GarageYrBlt', '1stFlrSF', 'TotRmsAbvGrd', 'GarageArea', # Correlated
        'PoolQC', 'MiscFeature', 'Alley', 'PoolArea' # High missing
    ]
    
    cat_missing_cols = [
        'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual',
        'GarageType', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond', 'MasVnrType'
    ]
    
    zero_impute_cols = [
        'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'BsmtFinSF1', 
        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'
    ]

    # Re-assemble pipeline
    final_pipeline = Pipeline([
        ('dropper', DropColumns(cols_to_drop)),
        ('lot_imputer', LotFrontageImputer()),
        ('mas_vnr_imputer', MasVnrAreaImputer()),
        ('cat_imputer', CategoricalImputer(cat_missing_cols)),
        ('zero_imputer', ZeroImputer(zero_impute_cols)),
        ('num_imputer', NumericImputer()), # Replaces generic SimpleImputer
        ('dummies', PandasGetDummies()),
        ('model', RandomForestRegressor(random_state=5))
    ])

    print("Training model...")
    final_pipeline.fit(X, y)
    print("Model trained.")

    # Save Defaults for App (Median/Mode for filling user inputs)
    # We need defaults for the *original* X columns (before drops/encoding)
    defaults = {}
    for col in X.columns:
        # Check if column is object or categorical
        if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
            defaults[col] = X[col].mode()[0] if not X[col].mode().empty else "None"
        else:
            defaults[col] = X[col].median()
            
    # Save Artifacts
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    joblib.dump(final_pipeline, os.path.join(artifacts_dir, 'pipeline.pkl'))
    joblib.dump(defaults, os.path.join(artifacts_dir, 'defaults.pkl'))
    
    print(f"Artifacts saved to {artifacts_dir}")

if __name__ == "__main__":
    train()
