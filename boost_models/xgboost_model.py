import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../data/final_prediction_dataset.csv')
data = pd.read_csv(data_path)

# Load the dataset
# data_path = '../data/final_prediction_dataset.csv'
# data = pd.read_csv(data_path)

#XGB wrapper to target failure of compatibility with different dependencies 
class CustomXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def __sklearn_tags__(self):
        return {
            "non_deterministic": False,
            "multioutput": False,
            "requires_positive_X": False,
            "poor_score": False,
        }

# Specify target and features
target_column = 'gross_adjusted'
X = data.drop(columns=[target_column])
y = data[target_column]


# Check for NaN or infinite values in the target
y = y.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
y = y.dropna()  # Drop rows with NaN values in the target


# Ensure X matches the filtered y by using y's index
X = X.loc[y.index]


# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()


# Preprocess using pipelines
preprocessor = ColumnTransformer(
   transformers=[
       ('num', StandardScaler(), numerical_cols),
       ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
   ]
)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Build pipeline
model_pipeline = Pipeline([
   ('preprocessor', preprocessor),
   ('regressor', CustomXGBRegressor(n_estimators=100, random_state=42))
])


# Train the model
model_pipeline.fit(X_train, y_train)


# Evaluate
y_pred = model_pipeline.predict(X_test)
# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse:.2f}')


from sklearn.metrics import r2_score


# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2:.2f}')
