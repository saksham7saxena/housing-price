# Housing Price Prediction

This project implements machine learning models to predict residential housing prices using the [Ames Housing dataset](https://www.kaggle.com/c/home-data-for-ml-course). The goal is to predict the final sale price of each home based on 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa.

## Project Overview

The project demonstrates a complete machine learning workflow:
1.  **Data Loading & Cleaning**: Handling missing values, outliers, and data types.
2.  **Feature Engineering**: 
    - Imputing missing values for features like `LotFrontage` and `MasVnrArea`.
    - Handling categorical variables using One-Hot Encoding.
    - Dropping highly correlated or redundant features to reduce multicollinearity.
3.  **Modeling**: Implementing and comparing two regression models:
    - **Linear Regression**: A baseline linear approach.
    - **Random Forest Regressor**: An ensemble learning method for better predictive performance.
4.  **Evaluation**: Using Mean Absolute Error (MAE) to evaluate model performance on a validation set.

## Dataset

The dataset contains training and testing data with the following key characteristics:
-   **Train**: 1460 entries, 81 columns (including Target `SalePrice`).
-   **Test**: 1459 entries, 80 columns.
-   **Features**: Includes numerical (e.g., `LotArea`, `YearBuilt`) and categorical (e.g., `Neighborhood`, `HouseStyle`) variables.

## Key Implementation Details

### Data Preprocessing
-   **Outlier Removal**: Removed extreme outliers in `LotFrontage`, `LotArea`, and `GrLivArea` to improve model stability.
-   **Feature Selection**: Dropped columns with >90% missing values (e.g., `PoolQC`, `MiscFeature`) and highly correlated features (e.g., `GarageYrBlt`, `1stFlrSF`).
-   **Log Transformation**: Applied `np.log()` to the target variable `SalePrice` to normalize its distribution, and `np.exp()` to inverse transform predictions.

## Model Performance

We evaluated the models using Mean Absolute Error (MAE) on a validation split:

| Model | Validation MAE |
| :--- | :--- |
| **Linear Regression** | ~$15,486 |
| **Random Forest** | ~$17,442 |

*Note: Results may vary slightly due to random states and environment differences.*

## Project Structure

```text
housing-price/
├── data/
│   ├── train.csv       # Training dataset
│   └── test.csv        # Testing dataset
├── src/
│   └── main.py         # Main script for Training and Prediction
├── requirements.txt    # Python dependencies
└── README.md           # Project Documentation
```

## Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/saksham7saxena/housing-price.git
    cd housing-price
    ```

2.  **Create and Activate a Virtual Environment** (Optional but recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main analysis script:

```bash
python src/main.py
```

This will:
-   Load and clean the data.
-   Train the models.
-   Print the validation MAE.
-   Generate a `submission.csv` file containing price predictions for the test dataset.

## Future Improvements
-   Implement Gradient Boosting models (XGBoost, LightGBM) for better accuracy.
-   Perform Hyperparameter Tuning using GridSearch or RandomizedSearch.
-   Explore more advanced Feature Engineering techniques.
