# Used Car Price Prediction - Kaggle Competition

Link: https://www.kaggle.com/competitions/playground-series-s4e9/data

## Overview
This project aims to predict the prices of used cars based on various attributes such as brand, model, manufacturing year, mileage, fuel type, engine specifications, and more. The predictions are evaluated based on Root Mean Squared Error (RMSE) in the Kaggle competition.

## Dataset
The dataset includes the following features:
- **Brand & Model**: Vehicle's brand and model.
- **Model Year**: Year of manufacturing, crucial for depreciation.
- **Mileage**: Indicator of the car's wear and tear.
- **Fuel Type**: Gasoline, Diesel, Hybrid, Electric, etc.
- **Transmission**: Automatic, Manual, or others.
- **Engine Specifications**: Engine size, horsepower, and cylinder count.
- **Accident History**: Whether a vehicle has been in an accident.
- **Clean Title**: Whether the vehicle has a clean title, affecting resale value.
- **Price**: The target variable in the training dataset.

The competition dataset consists of two files:
- **train.csv**: Contains the features along with the target (price) for training.
- **test.csv**: Contains only features for prediction (price is to be predicted).

## Modeling

### 1. Data Preprocessing
Before selecting a model, we carefully preprocessed the dataset to ensure that it was ready for training. Preprocessing steps included:

- **Handling Missing Data**: Missing values were filled based on logical assumptions or median values. Categorical columns with rare labels were grouped as 'Rare'.
- **Feature Extraction**: Several important features were derived from existing columns:
  - **Horsepower**: Extracted from the `engine` column (e.g., "172.0HP").
  - **Engine Size**: Extracted in liters from the `engine` column (e.g., "1.6L").
  - **Cylinders**: Parsed from the `engine` type or specific values (e.g., "V6", "I4").
  - **Fuel Type**: Extracted from the `engine` and `fuel_type` columns.
  - **Transmission Type**: Classified as `Automatic`, `Manual`, or `CVT` from the `transmission` column.
  - **Luxury & Top 10 Brands**: Binary features for `luxurious_brands` and `top_10_brands` were created to indicate premium and popular brands, respectively.

- **Data Transformation**: The numerical and categorical features were handled appropriately:
  - **Numerical Features**: Mileage, engine size, horsepower, and rank were kept as numeric variables.
  - **Categorical Features**: Converted into categories using Label Encoding for models like LightGBM, which can handle categorical data natively.

### 2. Model Selection and Baseline
We initially tried several models to set a baseline for prediction accuracy. The main models tested include:

1. **XGBoost Regressor**: Gradient boosting model for high-performance predictions.
2. **Random Forest Regressor**: Ensemble learning method using decision trees.
3. **LightGBM Regressor**: A fast and efficient gradient boosting model optimized for large datasets.
4. **CatBoost Regressor**: Gradient boosting model optimized for categorical features.
5. **Support Vector Regressor (SVR)**: Support Vector Machines applied to regression tasks.

### Model Performance
The performance of each model was evaluated using RMSE on the validation set:

| Model                 | RMSE        |
|-----------------------|-------------|
| **XGBoost**            | 68,873      |
| **Random Forest**      | 69,239      |
| **LightGBM**           | 68,559      |
| **CatBoost**           | 68,651      |
| **Support Vector**     | 74,412      |

Among these models, **LightGBM** and **CatBoost** performed the best, providing the lowest RMSE scores.


### 3. Model Optimization with LightGBM
Given its strong baseline performance, we chose **LightGBM** as the primary model. We applied the following steps to improve the model:

### Hyperparameter Tuning:
We utilized **Optuna** for hyperparameter tuning to optimize the performance of the LightGBM model. This involved adjusting key parameters such as:

- `n_estimators`
- `num_leaves`
- `max_depth`
- `learning_rate`
- `lambda_l1` and `lambda_l2`
- `subsample` and `colsample_bytree`

### Best Hyperparameters:
After 30 trials, the optimal parameters were found to be:
```yaml
- n_estimators: 1002
- num_leaves: 84
- max_depth: 21
- learning_rate: 0.0036
- subsample: 0.964
- colsample_bytree: 0.401
- lambda_l1: 5.16e-6
- lambda_l2: 0.00098
- max_bin: 667
```
### Model Performance:

The optimized LightGBM model achieved an RMSE of 65,722.60 on the validation set, improving from the initial baseline RMSE of 66,860.18

## Conclusion
In this project, we successfully built a used car price prediction model using various machine learning techniques. Through feature engineering, model selection, and hyperparameter tuning, we significantly improved the model's predictive power.

Key Highlights:
- **Data Preprocessing and Feature Extraction were vital to improving model performance.
- **LightGBM was chosen as the primary model due to its efficiency and performance with large-scale datasets.
- **Hyperparameter tuning using Optuna helped further refine the model, achieving a final RMSE of 65,722.60.
- **The insights from Feature Importance provided valuable understanding of which factors drive used car prices.
  
Overall, the project demonstrates a robust approach to predicting car prices and offers opportunities for further improvements through advanced ensemble methods and deeper feature exploration.


