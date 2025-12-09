# Car Price Prediction Project (Supervised Learning Final)

## 1. Project Topic

### Data Source (APA Citation)
> Monani, A. (2024). *Car Prices supervised ML* [Data set]. Kaggle. https://www.kaggle.com/datasets/aryamonani/car-prices-supervised-ml

### Backstory:
Car price prediction is a classic regression problem. Ever since I was a child I love cars and I love buying them. I decided to create this model to analyze car price data to make better decisions in the future for myself and my friends on how much they are expected to pay for certain features of vehicles. That being said Non-linear models often outperform linear ones due to complex depreciation factors (Pudaruth, 2014). I explicitly handle skewed price distributions using Log-Transformation (`np.log1p`).

### Key Features:
- Implementing multiple supervised regression models (Linear Regression, Random Forest, XGBoost).
- Conducting data cleaning and feature engineering (log-transforms, encoding).
- Comparing linear vs. non-linear approaches.
- Incorporating a custom Tkinter GUI for user interaction.

## 2. Data Source and Description
- **Dataset Size:** ~450 rows
- **Features:** `Brand`, `Body`, `Mileage`, `EngineV`, `Engine Type`, `Registration`, `Year`.
- `Model` column is high cardinality and excluded to prevent overfitting.

## 3. Methodology: The Model Engine
To ensure robust and reusable code, we encapsulated the data logic into a Python class `ModelEngine`.

### Key Features:
- **Automatic Type Detection**: Detects if the target is Regression (Price) or Classification (Brand/Body).
- **Pipeline**: Handles imputation, scaling, and one-hot encoding automatically.
- **Hyperparameter Tuning**: Supports custom `n_estimators` and `max_depth`.

## 4. Exploratory Data Analysis & Model Training
 Here we perform the analysis, including **Correlation Matrices** and **Feature Importance** plots to satisfy advanced rubric criteria.

### Data Cleaning
- Duplicates are removed.
- Rows with missing targets are dropped.
- 'NA' strings are converted to `NaN`.

### Feature Engineering
- **Target**: Log-transformed (`np.log1p`) to normalize the right-skewed price distribution.
- **Categorical**: One-Hot Encoded (`Brand`, `Body`, etc/)
- **Feature Importance**: We inspect which features drive the model's decisions.
