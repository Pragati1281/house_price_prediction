# 🏠 House Price Prediction

A machine learning project to predict house sale prices using the **House Prices: Advanced Regression Techniques** dataset from Kaggle.

The project covers exploratory data analysis, missing-value handling, categorical encoding, model comparison, hyperparameter tuning, and final Kaggle submission.

## 📌 Problem Statement

The goal of this project is to predict the **SalePrice** of residential properties based on features such as:

* Overall Quality
* Living Area
* Garage information
* Basement information
* Neighborhood
* Year Built
* Foundation
* And other property characteristics

## 📊 Dataset

Dataset: **House Prices: Advanced Regression Techniques**

The dataset contains information about residential properties in Ames, Iowa.

* Training samples: **1,460**
* Kaggle test samples: **1,459**
* Target variable: **SalePrice**

## 🔍 Exploratory Data Analysis

During EDA, I analyzed:

* Numerical feature correlations with `SalePrice`
* Important categorical features
* Neighborhood-wise average prices
* Foundation-wise average prices
* Sale condition and yearly price patterns
* Missing-value distribution

Some of the strongest numerical relationships with `SalePrice` were:

* `OverallQual`
* `GrLivArea`
* `GarageCars`
* `GarageArea`
* `TotalBsmtSF`
* `1stFlrSF`
* `FullBath`
* `YearBuilt`

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### Missing Values

Some missing values in the dataset represented the **absence of a feature**, such as:

* No garage
* No basement
* No pool
* No fireplace
* No alley access

These categorical missing values were replaced with `"None"`.

Remaining numerical missing values were handled using **median imputation**, while categorical missing values were handled using the most frequent value.

### Categorical Encoding

Categorical features were converted into numerical features using:

**OneHotEncoder**

with:

```python
handle_unknown='ignore'
```

### ColumnTransformer

A `ColumnTransformer` was used to apply different preprocessing to numerical and categorical features.

The final preprocessing pipeline converted the original features into **301 model-ready features**.

## 🤖 Models Tried

Several regression models were evaluated:

1. Linear Regression
2. Linear Regression with log-transformed target
3. Ridge Regression
4. Random Forest Regressor
5. Tuned Random Forest Regressor
6. XGBoost Regressor
7. Tuned XGBoost Regressor

## ⚙️ Hyperparameter Tuning

`RandomizedSearchCV` with 5-fold cross-validation was used to tune Random Forest and XGBoost.

The final XGBoost model used the following important parameters:

```python
{
    'subsample': 0.8,
    'n_estimators': 700,
    'min_child_weight': 5,
    'max_depth': 4,
    'learning_rate': 0.05,
    'gamma': 0,
    'colsample_bytree': 0.7
}
```

## 🧪 Feature Engineering

I also experimented with additional features:

* `TotalSF`
* `TotalBathrooms`
* `HouseAge`

These features were evaluated using the same validation setup.

Interestingly, they did **not improve the validation performance**, so the original feature set was retained for the final model.

## 🏆 Final Model

The final model selected was **Tuned XGBoost Regressor**.

### Validation Performance

| Metric |     Score |
| ------ | --------: |
| MAE    | 15,265.42 |
| RMSE   | 24,451.51 |
| R²     |    0.9221 |
| MAPE   |     9.30% |

The model achieved an **R² of approximately 92.2%** on the held-out validation set.

## 🏅 Kaggle Result

The final model was used to generate predictions for Kaggle's unseen test dataset.

**Kaggle Score: `0.13137`**

The predictions were submitted through the Kaggle competition.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Jupyter Notebook
* Kaggle

## 📁 Project Structure

```text
house-price-prediction/
│
├── house_price_prediction.ipynb
├── submission.csv
├── requirements.txt
└── README.md
```

## 📚 Key Learnings

Through this project, I practiced:

* Exploratory Data Analysis
* Feature selection
* Missing-value imputation
* One-hot encoding
* `ColumnTransformer`
* Machine learning pipelines
* Train-validation splitting
* Regression evaluation metrics
* Random Forest
* XGBoost
* Hyperparameter tuning with `RandomizedSearchCV`
* Feature engineering
* Kaggle competition workflow
* Creating and submitting Kaggle predictions

## 🚀 Future Improvements

Possible future improvements include:

* More advanced feature engineering
* Target transformation and blending/ensemble methods
* Cross-validation based model comparison
* More extensive hyperparameter optimization
* Experimenting with LightGBM or CatBoost
* Further analysis of outliers and skewed features
