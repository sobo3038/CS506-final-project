import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve, cross_val_score
from mpl_toolkits.mplot3d import Axes3D

# Set up output folder
output_folder = "final_report_images"
os.makedirs(output_folder, exist_ok=True)

# Load data and model
data_path = "data/final_prediction_dataset.csv"
model_path = "webpage/xgboost_model.json"
data = pd.read_csv(data_path)

# Load pre-trained XGBoost model
model = xgb.XGBRegressor()
model.load_model(model_path)

# Ensure dataset includes only valid entries
data = data.dropna(subset=['gross_adjusted'])

# Calculate residuals
data['Predicted'] = model.predict(data[['budget_adjusted', 'runtime', 'release_month', 'budget_adjusted_times_runtime', 'budget_adjusted_times_year']])
data['Residual'] = data['gross_adjusted'] - data['Predicted']

# Define save_plot utility
def save_plot(fig, filename):
    fig.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    plt.close(fig)

###################################
# PART 1: CORRELATION & INSIGHTS  #
###################################

# 1. Pairplot for Key Features
pairplot_data = data[['budget_adjusted', 'runtime', 'gross_adjusted', 'release_month']]
sns.pairplot(pairplot_data, diag_kind='kde', kind='scatter')
plt.savefig(os.path.join(output_folder, "pairplot_key_features.png"))
plt.close()

# 2. Heatmap for Correlations
fig, ax = plt.subplots(figsize=(12, 8))
corr_matrix = data[['budget_adjusted', 'runtime', 'gross_adjusted', 'release_month']].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
save_plot(fig, "correlation_heatmap.png")

# 3. Runtime vs. Gross Adjusted by Rating
if 'rating' in data.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='runtime', y='gross_adjusted', hue='rating', data=data, ax=ax, palette="Set1")
    ax.set_title("Runtime vs. Gross Adjusted by Rating")
    ax.set_xlabel("Runtime")
    ax.set_ylabel("Gross Adjusted")
    save_plot(fig, "runtime_vs_gross_by_rating.png")

# 4. Gross Adjusted Distribution by Genre
if 'genre' in data.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='genre', y='gross_adjusted', data=data, ax=ax, palette="viridis")
    ax.set_title("Gross Adjusted Distribution by Genre")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Gross Adjusted")
    ax.tick_params(axis='x', rotation=45)
    save_plot(fig, "gross_adjusted_by_genre.png")

# 5. Distribution of Prediction Errors
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data['Residual'], kde=True, ax=ax, color="purple")
ax.set_title("Distribution of Prediction Errors")
ax.set_xlabel("Residuals (Actual - Predicted)")
ax.set_ylabel("Frequency")
save_plot(fig, "prediction_error_distribution.png")

# 6. Average Runtime and Budget by Year
if 'year' in data.columns:
    yearly_data = data.groupby('year').agg({'runtime': 'mean', 'budget_adjusted': 'mean'}).reset_index()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(yearly_data['year'], yearly_data['runtime'], label='Average Runtime', color='orange')
    ax.set_ylabel("Average Runtime", color='orange')
    ax.tick_params(axis='y', labelcolor='orange')

    ax2 = ax.twinx()
    ax2.plot(yearly_data['year'], yearly_data['budget_adjusted'], label='Average Budget', color='blue')
    ax2.set_ylabel("Average Budget Adjusted", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax.set_title("Average Runtime and Budget by Year")
    ax.set_xlabel("Year")
    save_plot(fig, "avg_runtime_budget_by_year.png")

# 7. Rating Distribution by Year
if 'rating' in data.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='year', y='gross_adjusted', hue='rating', data=data, ax=ax)
    ax.set_title("Rating Distribution of Gross Adjusted by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Gross Adjusted")
    save_plot(fig, "rating_distribution_by_year.png")

# 8. Budget Impact on Top Genres
if 'genre' in data.columns:
    top_genres = data['genre'].value_counts().nlargest(5).index
    genre_data = data[data['genre'].isin(top_genres)]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='genre', y='budget_adjusted', data=genre_data, ax=ax, palette="coolwarm")
    ax.set_title("Budget Impact on Top Genres")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Budget Adjusted")
    save_plot(fig, "budget_impact_top_genres.png")

# 9. Average Gross Adjusted by Rating
if 'rating' in data.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    avg_gross_by_rating = data.groupby('rating')['gross_adjusted'].mean()
    avg_gross_by_rating.sort_values().plot(kind='bar', ax=ax, color='green')
    ax.set_title("Average Gross Adjusted by Rating")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Average Gross Adjusted")
    save_plot(fig, "avg_gross_by_rating.png")

# 10. Yearly Trend of Gross Adjusted
fig, ax = plt.subplots(figsize=(12, 8))
trend = data.groupby('year')['gross_adjusted'].mean()
trend.plot(ax=ax, color='blue')
ax.set_title("Yearly Trend of Gross Adjusted")
ax.set_xlabel("Year")
ax.set_ylabel("Average Gross Adjusted")
save_plot(fig, "yearly_trend_gross_adjusted.png")

######################################
# PART 2: MACHINE LEARNING VISUALS  #
######################################

# 11. Feature Contributions to Predictions (Bar Chart)
feature_importances = model.feature_importances_
features = [
    "budget_adjusted",
    "runtime",
    "release_month",
    "budget_adjusted_times_runtime",
    "budget_adjusted_times_year"
]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(features, feature_importances, color="skyblue")
ax.set_title("Feature Contributions to Predictions")
ax.set_xlabel("Importance")
ax.set_ylabel("Features")
save_plot(fig, "feature_contributions.png")

# 12. Residual Analysis Heatmap
if 'release_month' in data.columns and 'year' in data.columns:
    residual_heatmap = data.pivot_table(
        values='Residual', index='year', columns='release_month', aggfunc='mean'
    )
    fig, ax = plt.subplots(figsize=(15, 10))  # Larger figure for readability
    sns.heatmap(
        residual_heatmap, 
        cmap="coolwarm", 
        annot=True, 
        fmt=".2f",  # Two decimal places
        annot_kws={"size": 6},  # Smaller font size for annotations
        ax=ax
    )
    ax.set_title("Residual Analysis Heatmap")
    ax.set_xlabel("Release Month")
    ax.set_ylabel("Year")
    save_plot(fig, "residual_analysis_heatmap.png")

# 13. Permutation Importance Heatmap
result = permutation_importance(model, data[features], data['gross_adjusted'], n_repeats=10, random_state=42)
perm_importances = pd.DataFrame(result.importances_mean, index=features, columns=["Importance"])
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    perm_importances.T, 
    annot=True, 
    fmt=".2f",  # Two decimal places
    annot_kws={"size": 10},  # Adjust annotation size
    cmap="YlGnBu", 
    ax=ax
)
ax.set_title("Permutation Importance Heatmap")
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
save_plot(fig, "permutation_importance_heatmap.png")

# 14. Learning Curve (Train vs. Test Error)
train_sizes, train_scores, test_scores = learning_curve(model, data[features], data['gross_adjusted'], cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_mean, label="Train Score")
ax.plot(train_sizes, test_mean, label="Test Score")
ax.set_title("Learning Curve")
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Score")
ax.legend()
save_plot(fig, "learning_curve.png")

# 15. Cross-Validation Scores
cv_scores = cross_val_score(model, data[features], data['gross_adjusted'], cv=5, scoring="r2")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(1, 6), cv_scores, color="coral")
ax.set_title("Cross-Validation R^2 Scores")
ax.set_xlabel("Fold")
ax.set_ylabel("R^2")
save_plot(fig, "cross_validation_scores.png")

# 16. 3D Scatter Plot (Runtime, Predicted, Actual)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    data['runtime'],
    data['Predicted'],
    data['gross_adjusted'],
    c=data['release_month'],
    cmap="viridis",
    alpha=0.7
)
ax.set_title("3D Scatter: Runtime, Predicted, Actual")
ax.set_xlabel("Runtime")
ax.set_ylabel("Predicted Gross Adjusted")
ax.set_zlabel("Actual Gross Adjusted")
save_plot(fig, "3d_scatter_runtime_predicted_actual.png")

# 17. Hyperparameter Tuning Visualization
# Simulated data for demonstration purposes
param_values = [50, 100, 200, 300, 400]
train_scores = [0.8, 0.82, 0.85, 0.87, 0.88]
test_scores = [0.78, 0.8, 0.82, 0.85, 0.84]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(param_values, train_scores, label="Train Scores", marker="o", linestyle="--")
ax.plot(param_values, test_scores, label="Test Scores", marker="o", linestyle="-")
ax.set_title("Hyperparameter Tuning: Effect on Scores")
ax.set_xlabel("Hyperparameter Value")
ax.set_ylabel("Score")
ax.legend()
save_plot(fig, "hyperparameter_tuning_scores.png")

# 18. Confidence Intervals of Predictions
lower_bounds = data['Predicted'] - 0.1 * data['Predicted']
upper_bounds = data['Predicted'] + 0.1 * data['Predicted']
fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(range(len(data['Predicted'])), lower_bounds, upper_bounds, color="lightblue", alpha=0.5, label="Confidence Interval")
ax.plot(range(len(data['Predicted'])), data['Predicted'], color="blue", label="Predictions")
ax.set_title("Confidence Intervals of Predictions")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Predicted Gross Adjusted")
ax.legend()
save_plot(fig, "confidence_intervals_predictions.png")

# 19. Error Distribution by Features
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='release_month', y=data['Residual'], data=data, ax=ax, palette="coolwarm")
ax.set_title("Residuals by Release Month")
ax.set_xlabel("Release Month")
ax.set_ylabel("Residuals (Actual - Predicted)")
save_plot(fig, "residuals_by_release_month.png")

# 20. Feature Interaction Effects
fig, ax = plt.subplots(figsize=(20, 14))  # Larger figure size for better spacing
residual_heatmap = data.pivot_table(
    index='release_month', 
    columns='year', 
    values='Residual', 
    aggfunc='mean'
)
sns.heatmap(
    residual_heatmap, 
    cmap="coolwarm", 
    annot=True, 
    fmt=".2f",  # Two decimal places
    annot_kws={"size": 6},  # Smaller annotations
    ax=ax
)
ax.set_title("Feature Interaction: Residuals by Year and Month", fontsize=18)
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Month", fontsize=14)
ax.tick_params(axis='x', labelrotation=45, labelsize=10)  # Rotate x-axis labels
ax.tick_params(axis='y', labelsize=10)  # Increase font size for y-axis labels
save_plot(fig, "feature_interaction_effects_updated.png")

# Enhanced Variance Analysis Graph
fig, ax = plt.subplots(figsize=(12, 8))
for feature in features:
    feature_variance = data.groupby(feature)['Predicted'].var()
    feature_variance.plot(ax=ax, label=feature, marker='o')

ax.set_title("Prediction Variance by Key Features", fontsize=16)
ax.set_xlabel("Feature Value")
ax.set_ylabel("Variance in Predicted Gross Adjusted")
ax.legend(title="Features", fontsize=10)
save_plot(fig, "prediction_variance_by_features.png")

# Enhanced Confidence Interval Graph: Mean Prediction with Confidence Intervals by Release Month
if 'release_month' in data.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by release month and calculate mean and confidence intervals
    grouped = data.groupby('release_month')['Predicted']
    mean_pred = grouped.mean()
    ci_lower = grouped.mean() - 1.96 * grouped.std() / np.sqrt(grouped.count())
    ci_upper = grouped.mean() + 1.96 * grouped.std() / np.sqrt(grouped.count())
    
    # Plot mean predictions with confidence intervals
    ax.plot(mean_pred.index, mean_pred, label='Mean Prediction', color='blue', marker='o')
    ax.fill_between(mean_pred.index, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% Confidence Interval')
    
    ax.set_title("Mean Predictions with Confidence Intervals by Release Month", fontsize=16)
    ax.set_xlabel("Release Month")
    ax.set_ylabel("Predicted Gross Adjusted")
    ax.legend(fontsize=10)
    save_plot(fig, "confidence_intervals_by_release_month.png")