# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- Step 1: Set Path to Feature Engineering Script ---
# Load the adjusted dataset
data = pd.read_csv('../data/movies_adjusted.csv')  # Load the original dataset

# --- Step 2: Run Feature Engineering ---
# Drop non-predictive and dependent columns
data = data.drop(columns=['gross', 'budget', 'name', 'score', 'votes'], errors='ignore')

# Extract date only from 'released' and parse it
data['released'] = data['released'].str.extract(r'(\w+ \d{1,2}, \d{4})')[0]
data['released'] = pd.to_datetime(data['released'], format='%B %d, %Y', errors='coerce')
data['release_year'] = data['released'].dt.year
data['release_month'] = data['released'].dt.month

# Set target variable
target = 'gross_adjusted'

# Drop the original 'released' column after extracting date features
data = data.drop(columns=['released'], errors='ignore')

# Mean Encoding for Categorical Variables
categorical_features = ['rating', 'genre', 'director', 'writer', 'star', 'country', 'company']
categorical_corrs = {}

for feature in categorical_features:
    # Calculate the mean target value for each category and use it to encode the feature
    data[f'{feature}_encoded'] = data.groupby(feature)[target].transform('mean')
    # Calculate the correlation with the target
    correlation = data[[f'{feature}_encoded', target]].corr().iloc[0, 1]
    categorical_corrs[feature] = correlation

# Drop the original categorical features
data = data.drop(columns=categorical_features, errors='ignore')

# Handle missing values
data = data.fillna(0)
print("Feature Engineering completed!")

# --- Step 3: Prepare Data for Modeling ---
# Drop any columns that are not needed for the model
features = data.drop(columns=['gross_adjusted'], errors='ignore')
target = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print("Dataset split!")

# --- Step 4: Model Training and Evaluation ---
# Initialize the Decision Tree model
decision_tree_model = DecisionTreeRegressor(random_state=42)
print("Training Decision Tree model...")
decision_tree_model.fit(X_train, y_train)  # Train the model
dt_predictions = decision_tree_model.predict(X_test)  # Make predictions
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))  # Calculate RMSE
dt_r2 = r2_score(y_test, dt_predictions)  # Calculate R² score
print("Finished Decision Tree model!")

# Initialize the Random Forest model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Training Random Forest model...")
random_forest_model.fit(X_train, y_train)  # Train the model
rf_predictions = random_forest_model.predict(X_test)  # Make predictions
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))  # Calculate RMSE
rf_r2 = r2_score(y_test, rf_predictions)  # Calculate R² score
print("Finished Random Forest model!")

# Output the performance metrics
print("Decision Tree Model Performance:")
print(f"RMSE: {dt_rmse:.2f}")
print(f"R²: {dt_r2:.2f}\n")

print("Random Forest Model Performance:")
print(f"RMSE: {rf_rmse:.2f}")
print(f"R²: {rf_r2:.2f}")
