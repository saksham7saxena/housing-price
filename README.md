# Housing Price Prediction System

This project implements a robust machine learning pipeline and an interactive Streamlit web application to predict residential housing prices using the [Ames Housing dataset](https://www.kaggle.com/c/home-data-for-ml-course).

## 🚀 Live Demo
Run the app locally to interact with the model:
```bash
streamlit run app.py
```

## 🏗️ Technical Architecture

The system is built on a modular `scikit-learn` pipeline that ensures consistency between training and inference (prediction).

### 1. Data Pipeline (`src/train.py`)
The training script orchestrates the following ETL and preprocessing steps:

#### A. Data Cleaning & Feature Engineering
Custom transformers were implemented to replicate specific domain logic:
*   **`DropColumns`**: Removes features with >90% missing values (e.g., `PoolQC`) and highly detected multicollinearity (e.g., `GarageYrBlt` vs `YearBuilt`, `1stFlrSF` vs `TotalBsmtSF`).
*   **`LotFrontageImputer`**: Imputes missing `LotFrontage` using the square root of `LotArea` (assumes square lots), capturing geometric relationships.
*   **`MasVnrAreaImputer`**: Fills missing masonry veneer areas with the mean of the training set.
*   **`CategoricalImputer`**: Fills missing categorical values (e.g., `FireplaceQu`, `GarageType`) with `"None"`, treating absence as a valid category information.
*   **`ZeroImputer`**: Fills missing numeric structural features (e.g., `BsmtFinSF1`) with `0` (implying absence of basement).

#### B. Preprocessing & Encoding
*   **Numeric Features**: Missing values in remaining numeric columns are imputed using the **Median** strategy to be robust against outliers.
*   **Categorical Features**: One-Hot Encoding is applied via a custom `PandasGetDummies` transformer that aligns training and test columns, ensuring the model always receives a fixed feature vector structure.

#### C. Target Transformation
*   The target variable `SalePrice` is **Log-Transformed** (`np.log`) prior to training. This normalizes the skewed price distribution, improving the performance of regression models.
*   Predictions are inverse-transformed (`np.exp`) to return values in dollars.

### 2. Model Configuration
*   **Algorithm**: Random Forest Regressor
*   **Parameters**: `random_state=5` (fixed for reproducibility).
*   **Performance**: The model achieves a **Mean Absolute Error (MAE)** of approximately **$17,442** on the validation set.

### 3. Streamlit Application (`app.py`)
The web interface decouples the UI from the model logic:
*   **Artifact Loading**: Loads the pre-trained `pipeline.pkl` and `defaults.pkl` (feature modes/medians) using `joblib`.
*   **Dynamic Input Handling**: The app exposes ~10 high-impact features (e.g., Neighborhood, Quality, Area). All hidden features required by the pipeline are automatically filled with values from `defaults.pkl`.
*   **Inference**:
    1.  User inputs are merged with default values.
    2.  Data is passed through the full transformation pipeline.
    3.  Model predicts log-price, which is converted to USD.
*   **Interpretability**: Displays top feature drivers using Random Forest feature importances.

## 📂 Project Structure

```text
housing-price/
├── artifacts/
│   ├── pipeline.pkl    # Serialized sklearn pipeline (Transformers + Model)
│   └── defaults.pkl    # Dictionary of median/mode values for all features
├── data/               # Raw CSV datasets
├── src/
│   ├── train.py        # Pipeline definition, training, and artifact generation
│   └── main.py         # (Legacy) exploratory script
├── app.py              # Streamlit frontend
└── requirements.txt    # Project dependencies
```

## 🛠️ Setup & Usage

### Prerequisites
*   Python 3.8+
*   Virtual Environment (Recommended)

### Installation
```bash
git clone https://github.com/saksham7saxena/housing-price.git
cd housing-price
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Operations
*   **Train Model**: `python src/train.py` (Regenerates artifacts)
*   **Run App**: `streamlit run app.py` (Starts web server)
