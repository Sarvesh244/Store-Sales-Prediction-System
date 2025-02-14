# Store Sales Prediction System

## Introduction
Sales prediction is an important part of modern business intelligence. It can be a complex problem, especially in the case of lack of data, missing data, and the presence of outliers. Sales prediction is rather a regression problem than a time series problem. Practice shows that the use of regression approaches can often give us better results compared to time series methods. Machine-learning algorithms make it possible to find patterns in the time series. Some of the most popular are tree-based machine-learning algorithms.

## Conclusion
Adaboost gave better scores and variance sometimes versus Random Trees. But XGBoost gave great results with the existing dataset. Interestingly, the features that XGBoost chose did not have the same ranking as other models. Average sales per month or per day of the week did not give an increase, most likely due to extrapolating data where there wasn’t any in the test set, leading to overfitting. By this, the prediction is being made for the stores.

## Project Structure

- `Store Sales Prediction System.ipynb`: Jupyter notebook containing the code for data preprocessing, feature engineering, model training, and evaluation.
- `sales.csv`: CSV file containing sales data.
- `store.csv`: CSV file containing store information.
- `test_v2.csv`: CSV file containing test data.
- `train_v2.csv`: CSV file containing training data.

## Requirements

- Python 3.x
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Setup

1. Clone the repository.
2. Install the required packages:
    ```sh
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost
    ```
3. Open `Store Sales Prediction System.ipynb` in Jupyter Notebook.

## Usage

1. Load the data:
    ```python
    types = {'StateHoliday': np.dtype(str)}
    train = pd.read_csv("train_v2.csv", parse_dates=[2], nrows=66901, dtype=types)
    store = pd.read_csv("store.csv")
    ```
2. Preprocess the data and build features:
    ```python
    features = build_features(train, store)
    ```
3. Train and evaluate the model:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(features[X], features['Sales'], test_size=0.15, random_state=10)
    ```

## Functions

- `build_features(train, store)`: Preprocesses the data and builds features for the model.
- `rmspe(y, y_hat)`: Calculates the Root Mean Square Percentage Error.
- `score(model, X_train, y_train, y_test, y_hat)`: Evaluates the model using cross-validation and RMSPE.
- `plot_importance(model)`: Plots the feature importance of the model.

## License

This project is licensed under the MIT License.