# California Housing Price Prediction using XGBoost

## Project Overview
This project aims to predict housing prices in California using data from the California Housing dataset. The model leverages the XGBoost Regressor for predictive analysis, evaluating performance based on R-squared and Mean Absolute Error (MAE) metrics.

## Project Structure
The project includes:
- **Data Preprocessing**: Handling missing values, splitting the data into training and testing sets.
- **Exploratory Data Analysis (EDA)**: Visualizing correlations between features using a heatmap.
- **Model Training**: Training an XGBoost Regressor model on the training data.
- **Model Evaluation**: Comparing model predictions with actual values using R-squared and MAE metrics for both training and test sets.

## Tools and Libraries Used
- **Python** 
- **Pandas** 
- **Seaborn** 
- **Matplotlib** 
- **Scikit-learn** 
- **XGBoost**

## Performance Metrics
- **Training Data**:
   - R-squared: 0.9437
   - Mean Absolute Error (MAE): 0.1934
- **Test Data**:
   - R-squared: 0.8338
   - Mean Absolute Error (MAE): 0.3109

## Dataset
The dataset used in this project is the California Housing dataset, which can be fetched using the `fetch_california_housing()` function from `sklearn.datasets`.

## Project Setup and Usage

### Requirements
Install the required libraries by running the following command:
```bash
pip install -r requirements.txt
