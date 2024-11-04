# Predictive Analytics for Used Car Pricing

## Problem
The used car market is vast and complex, with various factors influencing car prices, including brand, model, year, mileage, and condition. Accurately predicting the price of used cars is crucial for buyers and sellers to make informed decisions, ultimately leading to fair transactions and a more efficient market.

## Solution
This project leverages machine learning techniques to build a regression model that predicts used car prices based on a rich dataset of car attributes. By employing advanced algorithms such as XGBoost, this solution outperforms traditional pricing methods by delivering higher accuracy and reliability in price estimation, thus reducing the gap between buyer expectations and seller offerings.

## Dataset
The dataset used in this project consists of 188,533 records of used cars with various attributes. Key columns include:
- **id**: Unique identifier for each car
- **brand**: Car brand
- **model**: Car model
- **model_year**: Year of manufacture
- **milage**: Mileage of the car
- **fuel_type**: Type of fuel used
- **engine**: Engine specifications
- **transmission**: Transmission type
- **ext_col**: Exterior color
- **int_col**: Interior color
- **accident**: Accident history
- **clean_title**: Title condition
- **price**: Target variable for prediction

**Source**: [Kaggle Playground Series - Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9/overview)

## Model Details
The model implemented is the **XGBoost Regressor**, a powerful gradient boosting framework that has proven effective in various regression tasks. The hyperparameters used for the model include:
- **Objective**: `reg:squarederror`
- **Number of estimators**: 100
- **Learning rate**: 0.1
- **Device**: `cuda:0` (for GPU acceleration)

## Evaluation
The model's performance is evaluated using Root Mean Squared Error (RMSE), which is a standard metric for regression tasks. The RMSE of the model on the validation set was **<insert your RMSE value>**, indicating the model's predictive accuracy.

## Citation
Walter Reade and Ashley Chow. Regression of Used Car Prices. [Kaggle](https://kaggle.com/competitions/playground-series-s4e9), 2024. Kaggle.
