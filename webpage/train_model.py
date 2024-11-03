import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('../data/final_prediction_dataset.csv')  # Adjust the path if necessary

# Drop rows with NaN values in the target column
data = data.dropna(subset=['gross_adjusted'])

# Define features and target variable
X = data[['budget_adjusted', 'runtime', 'release_month', 'budget_adjusted_times_runtime', 'budget_adjusted_times_year']]
y = data['gross_adjusted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Save the trained model in the 'webpage' folder as xgboost_model.json
model.save_model("xgboost_model.json")
print("Model saved as xgboost_model.json")
