import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('data/final_prediction_dataset.csv')

# Drop rows where the target (revenue) is NaN
data = data.dropna(subset=['gross_adjusted'])

# Define the target as movie revenue 
y = data['gross_adjusted']  
X = data.drop(columns=['gross_adjusted'])  # Remove target from features

# Select only numerical features for simplicity
X = X.select_dtypes(include=['number'])

# Impute missing values with the mean for numerical columns in features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)  # Update X with imputed values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Selection using SelectKBest to find the best features
k_best_selector = SelectKBest(score_func=f_regression, k='all')
X_train_selected = k_best_selector.fit_transform(X_train, y_train)
X_test_selected = k_best_selector.transform(X_test)

# Get feature scores and names
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': k_best_selector.scores_
}).sort_values(by='Score', ascending=False)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_selected, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Feature Scores:\n", feature_scores)
print("\nMean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Visualization

# Set up the visualization with three subplots
plt.figure(figsize=(18, 5))

# 1. Linear Regression Fit (Data points and Regression Line for the top feature, if any)
# For visualization purposes, we need to select a single feature; we'll pick the top-ranked feature
top_feature_index = feature_scores.index[0]
X_top_feature = X.iloc[:, top_feature_index]

# Scatter plot with regression line using the top feature
plt.subplot(1, 3, 1)
plt.scatter(X_top_feature, y, color="blue", label="Actual Revenue")
plt.plot(X_top_feature, model.predict(k_best_selector.transform(X)), color="red", linewidth=2, label="Regression Line")
plt.title("Linear Regression Fit")
plt.xlabel(f"Top Feature: {feature_scores.iloc[0, 0]}")
plt.ylabel("Revenue")
plt.legend()

# 2. Residual Plot for MSE
plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color="purple")
plt.hlines(0, y_pred.min(), y_pred.max(), color="red", linestyle="--")
plt.title(f"Residuals (MSE = {mse:.2f})")
plt.xlabel("Predicted Revenue")
plt.ylabel("Residuals")

# 3. R-squared Visualization (Overlayed Line with R² Annotation)
plt.subplot(1, 3, 3)
plt.scatter(X_top_feature, y, color="blue", label="Actual Revenue")
plt.plot(X_top_feature, model.predict(k_best_selector.transform(X)), color="red", linewidth=2, label=f"Regression Line (R² = {r2:.2f})")
plt.title("Model Fit Quality (R² Score)")
plt.xlabel(f"Top Feature: {feature_scores.iloc[0, 0]}")
plt.ylabel("Revenue")
plt.legend()

plt.tight_layout()
plt.show()
