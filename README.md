# Housing Price Prediction App

This project implements a machine learning model to predict housing prices using the [Ames Housing dataset](https://www.kaggle.com/c/home-data-for-ml-course). It features a **Streamlit Web App** for interactive predictions.

## Project Structure

```text
housing-price/
├── artifacts/          # Saved model pipeline and default values
├── data/               # Training and testing datasets
├── src/
│   ├── train.py        # Script to train model and save artifacts
│   └── main.py         # (Legacy) Original batch processing script
├── app.py              # Streamlit Web Application
├── requirements.txt    # Python dependencies
└── README.md           # Project Documentation
```

## Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/saksham7saxena/housing-price.git
    cd housing-price
    ```

2.  **Create and Activate a Virtual Environment**:
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

### 1. Run the Web App
To launch the interactive prediction tool:
```bash
streamlit run app.py
```
This will open the app in your browser at `http://localhost:8501`.

### 2. Re-Train the Model (Optional)
If you want to regenerate the model artifacts:
```bash
python src/train.py
```
This saves `pipeline.pkl` and `defaults.pkl` to the `artifacts/` folder.

## Model Details
- **Algorithm**: Random Forest Regressor
- **Validation MAE**: ~$17,442
- **Features**: The app exposes key features like Neighborhood, Living Area, Quality, and Year Built. Missing inputs are filled with dataset medians/modes.
