import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data_path = '../data/final_prediction_dataset.csv'
movies_metadata_df = pd.read_csv(data_path, low_memory=False)

# Print column names to verify
print("Columns in dataset:", movies_metadata_df.columns)

# Select necessary columns that match the CSV structure
selected_columns = ['budget_adjusted', 'genre', 'country', 'runtime', 'gross_adjusted']
missing_columns = [col for col in selected_columns if col not in movies_metadata_df.columns]

if missing_columns:
    raise KeyError(f"Missing columns in dataset: {missing_columns}")

movies_metadata_df = movies_metadata_df[selected_columns]

# Extract primary genre directly from 'genre' column
movies_metadata_df = movies_metadata_df.rename(columns={'genre': 'primary_genre'})

# Data cleaning
movies_metadata_df['budget_adjusted'] = pd.to_numeric(movies_metadata_df['budget_adjusted'], errors='coerce')
movies_metadata_df['runtime'] = pd.to_numeric(movies_metadata_df['runtime'], errors='coerce')
movies_metadata_df['gross_adjusted'] = pd.to_numeric(movies_metadata_df['gross_adjusted'], errors='coerce')
movies_metadata_df = movies_metadata_df.dropna()

# Feature engineering: One-hot encoding for categorical features
movies_metadata_encoded_df = pd.get_dummies(movies_metadata_df, columns=['country', 'primary_genre'], drop_first=True)

# Define features and target
X = movies_metadata_encoded_df.drop(columns=['gross_adjusted'])
y = movies_metadata_encoded_df['gross_adjusted']

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate and save the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Save the model and feature columns
joblib.dump(model, "linear_regression_model.pkl")
joblib.dump(X.columns, "feature_columns.pkl")
