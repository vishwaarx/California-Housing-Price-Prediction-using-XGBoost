
# California House Prices Prediction using XGBoost

This project demonstrates a machine learning model trained to predict California house prices using the **XGBoost Regressor** model and the California Housing dataset from Scikit-learn. The dataset includes a variety of features describing different aspects of residential homes in California.

## Table of Contents

- [Installation](#installation)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

## Installation

Clone the repository and install the dependencies.

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

## Project Overview

This project aims to predict the price of houses in California using the following steps:
1. Load and explore the dataset.
2. Preprocess the data and check for missing values.
3. Train an XGBoost Regressor model.
4. Evaluate the model performance.
5. Visualize the results.

## Dataset

The **California Housing dataset** from Scikit-learn is used in this project. It contains the following features:

- MedInc: Median income in block group
- HouseAge: Median house age in block group
- AveRooms: Average number of rooms
- AveBedrms: Average number of bedrooms
- Population: Block group population
- AveOccup: Average house occupancy
- Latitude: Block group latitude
- Longitude: Block group longitude

The target variable is the **Price** of the houses.

## Model Training

The project uses **XGBoost Regressor** for training the model. Below is a simplified workflow for training the model:

```python
# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Load and train the model
model = XGBRegressor()
model.fit(x_train, y_train)
```

## Evaluation

The model is evaluated using the **R-Squared score** and **Mean Absolute Error (MAE)**. For training data:

```python
# R squared error
r2_score_train = metrics.r2_score(y_train, training_data_prediction)

# Mean Absolute Error
mae_train = metrics.mean_absolute_error(y_train, training_data_prediction)
```

## Results

- **Training Data**:
  - R-Squared Score: `0.9436`
  - Mean Absolute Error: `0.1934`
  
- **Test Data**:
  - R-Squared Score: `0.8338`
  - Mean Absolute Error: `0.3109`

## Visualization

We visualize the model's performance by plotting the actual vs predicted house prices:

```python
plt.scatter(y_test, test_data_prediction)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs Predicted prices")
plt.show()
```

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
```

This final version includes the actual performance results for both the training and test datasets, as well as a structured explanation of the project. Replace the GitHub URL with your actual repository URL before publishing.
