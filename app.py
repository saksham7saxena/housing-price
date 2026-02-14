import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Define the custom transformers here so joblib can find them when loading
# (In a larger project, these should be in a shared module)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop
    def fit(self, X, y=None): return self
    def transform(self, X):
        cols = [c for c in self.cols_to_drop if c in X.columns]
        return X.drop(columns=cols)

class LotFrontageImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if 'LotFrontage' in X.columns and 'LotArea' in X.columns:
            X['LotFrontage'] = X['LotFrontage'].fillna(np.sqrt(X['LotArea']))
        return X

class MasVnrAreaImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if 'MasVnrArea' in X.columns: self.fill_value_ = X['MasVnrArea'].mean()
        else: self.fill_value_ = 0
        return self
    def transform(self, X):
        X = X.copy()
        if 'MasVnrArea' in X.columns: X['MasVnrArea'] = X['MasVnrArea'].fillna(self.fill_value_)
        return X

class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns: X[col] = X[col].fillna("None")
        return X

class ZeroImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns: X[col] = X[col].fillna(0)
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
        self.train_columns_ = pd.get_dummies(X).columns
        return self
    def transform(self, X):
        X_d = pd.get_dummies(X)
        X_d = X_d.reindex(columns=self.train_columns_, fill_value=0)
        return X_d

# --- App Logic ---

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠")

st.title("🏠 Ames Housing Price Predictor")
st.write("Enter the details of the house below to get an estimated sale price.")

# Load Artifacts
@st.cache_resource
def load_artifacts():
    artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
    pipeline = joblib.load(os.path.join(artifacts_dir, 'pipeline.pkl'))
    defaults = joblib.load(os.path.join(artifacts_dir, 'defaults.pkl'))
    return pipeline, defaults

try:
    pipeline, defaults = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run `python src/train.py` first.")
    st.stop()

# --- User Input Form ---

with st.form("house_details"):
    st.subheader("Property Basics")
    col1, col2 = st.columns(2)
    
    with col1:
        neighborhood = st.selectbox("Neighborhood", 
            ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
             'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
             'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
             'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
             'Blueste'])
             
        house_style = st.selectbox("House Style", 
            ['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'])
            
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
        year_built = st.number_input("Year Built", min_value=1870, max_value=2025, value=1990)

    with col2:
        gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=6000, value=1500)
        lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=100000, value=9000)
        bedroom_abv_gr = st.number_input("Bedrooms (Above Grade)", 0, 8, 3)
        full_bath = st.number_input("Full Bathrooms", 0, 4, 2)

    st.subheader("Features & Garage")
    col3, col4 = st.columns(2)
    
    with col3:
        garage_cars = st.number_input("Garage Cars", 0, 4, 2)
        total_bsmt_sf = st.number_input("Total Basement (sq ft)", 0, 6000, 1000)
        
    with col4:
        kitchen_qual = st.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa'])
        fireplace_qu = st.selectbox("Fireplace Quality", ['None', 'Ex', 'Gd', 'TA', 'Fa', 'Po'])

    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    # 1. Create DataFrame with Defaults
    input_data = defaults.copy()
    
    # 2. Update with User Inputs
    # Note: We must ensure the keys match exactly with dataset columns
    input_data['Neighborhood'] = neighborhood
    input_data['HouseStyle'] = house_style
    input_data['OverallQual'] = overall_qual
    input_data['YearBuilt'] = year_built
    input_data['GrLivArea'] = gr_liv_area
    input_data['LotArea'] = lot_area
    input_data['BedroomAbvGr'] = bedroom_abv_gr
    input_data['FullBath'] = full_bath
    input_data['GarageCars'] = garage_cars
    input_data['TotalBsmtSF'] = total_bsmt_sf
    input_data['KitchenQual'] = kitchen_qual
    # Handle "None" for fireplace properly if dataset uses NaN or specific string
    input_data['FireplaceQu'] = fireplace_qu if fireplace_qu != 'None' else np.nan 

    # Convert to DataFrame (single row)
    input_df = pd.DataFrame([input_data])
    
    # Ensure dtypes match (simple way is to infer)
    input_df = input_df.infer_objects()

    # 3. Predict
    with st.spinner("Calculating..."):
        try:
            log_prediction = pipeline.predict(input_df)[0]
            price = np.exp(log_prediction)
            
            # Confidence Interval (Approx +/- 10%)
            lower_bound = price * 0.9
            upper_bound = price * 1.1
            
            st.success(f"### Estimated Price: ${price:,.2f}")
            st.info(f"Price Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
            
            # Feature Importance Logic (Simplified for Random Forest)
            # Access the model step
            rf_model = pipeline.named_steps['model']
            
            # Get One-Hot Encoded Columns from the transformer
            dummies_step = pipeline.named_steps['dummies']
            feature_names = dummies_step.train_columns_
            
            # Get Feature Importances
            importances = rf_model.feature_importances_
            
            # Create DataFrame
            feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(3)
            
            st.subheader("Top Drivers of Price (Global)")
            for i, row in feat_imp.iterrows():
                st.write(f"- **{row['Feature']}**: {row['Importance']:.1%}")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.exception(e)

st.markdown("---")
st.caption("Built with Streamlit • Model: Random Forest")
