# Housing Price Prediction

This project implements a machine learning model to predict housing prices using the Ames Housing dataset. It uses Linear Regression and Random Forest models.

## Project Structure

- `data/`: Contains the training (`train.csv`) and testing (`test.csv`) datasets.
- `src/`: Contains the source code.
    - `main.py`: Main script to train the model and generate predictions.
- `requirements.txt`: List of Python dependencies.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/saksham7saxena/housing-price.git
    cd housing-price
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script to train the models and generate a submission file:

```bash
python src/main.py
```

This will:
1.  Load data from `data/`.
2.  Preprocess and clean the data.
3.  Train Linear Regression and Random Forest models.
4.  Print validation MAE for both models.
5.  Generate `submission.csv` with predictions for the test set.
