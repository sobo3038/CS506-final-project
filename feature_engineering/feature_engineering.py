import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Load the adjusted dataset
data = pd.read_csv('data/movies_adjusted.csv')

# Ensure we only work with adjusted monetary values and drop non-predictive columns
data = data.drop(columns=['gross', 'budget', 'name', 'score', 'votes'], errors='ignore')

# Extract `release_month` from the `released` column
data['released'] = pd.to_datetime(data['released'], errors='coerce')
data['release_month'] = data['released'].dt.month

# Set paths and target variable
target = 'gross_adjusted'
output_folder = "feature_engineering"
os.makedirs(output_folder, exist_ok=True)
suggestions_file = os.path.join(output_folder, 'suggestions.txt')

# Initialize lists to store calculated features and correlations
all_features = []
suggested_features = []

# Step 1: Heatmap for Original Numerical Features
numerical_data = data[['budget_adjusted', 'runtime', 'year', 'release_month', target]]
original_numerical_corr = numerical_data.corr()[[target]].drop(target).sort_values(by=target, ascending=False)

plt.figure(figsize=(6, 8))
sns.heatmap(original_numerical_corr, annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
plt.title('Correlation of Original Numerical Features with Adjusted Gross')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "original_numerical_features_correlation_heatmap.png"))
plt.close()

# Step 2: Heatmap for Original Categorical Features
# Convert each categorical feature to numeric (by encoding each category's average gross)
categorical_features = ['rating', 'genre', 'director', 'writer', 'star', 'country', 'company']
categorical_corrs = {}

for feature in categorical_features:
    data[f'{feature}_encoded'] = data.groupby(feature)[target].transform('mean')
    correlation = data[[f'{feature}_encoded', target]].corr().iloc[0, 1]
    categorical_corrs[feature] = correlation
    data.drop(columns=[f'{feature}_encoded'], inplace=True)

categorical_corr_df = pd.DataFrame.from_dict(categorical_corrs, orient='index', columns=['Correlation']).sort_values(by='Correlation', ascending=False)

plt.figure(figsize=(6, 8))
sns.heatmap(categorical_corr_df, annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
plt.title('Correlation of Original Categorical Features with Adjusted Gross')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "original_categorical_features_correlation_heatmap.png"))
plt.close()

# Step 3: Calculate Ratios and Interactions and their Correlations with gross_adjusted
numerical_features = numerical_data.columns.drop(target)
for (feature1, feature2) in combinations(numerical_features, 2):
    # Ratios
    if data[feature2].abs().min() > 1e-9:  # Avoid division by zero
        ratio_feature_name = f"{feature1}_to_{feature2}_ratio"
        data[ratio_feature_name] = data[feature1] / data[feature2]
        correlation = data[[target, ratio_feature_name]].corr().iloc[0, 1]
        all_features.append((ratio_feature_name, correlation))
        if abs(correlation) > 0.60:
            suggested_features.append((ratio_feature_name, correlation))
        data.drop(columns=[ratio_feature_name], inplace=True)

    # Interactions
    interaction_feature_name = f"{feature1}_times_{feature2}"
    data[interaction_feature_name] = data[feature1] * data[feature2]
    correlation = data[[target, interaction_feature_name]].corr().iloc[0, 1]
    all_features.append((interaction_feature_name, correlation))
    if abs(correlation) > 0.60:
        suggested_features.append((interaction_feature_name, correlation))
    data.drop(columns=[interaction_feature_name], inplace=True)

# Save Ratios and Interactions Heatmap
all_features_df = pd.DataFrame(all_features, columns=['Feature', 'Correlation']).set_index('Feature').sort_values(by='Correlation', ascending=False)

plt.figure(figsize=(6, len(all_features_df) // 4))
sns.heatmap(all_features_df, annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
plt.title('Correlation of All Ratios and Interactions with Adjusted Gross')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "ratios_interactions_correlation_heatmap.png"))
plt.close()

# Step 4: Mean Aggregations for Categorical Variables and their Correlations with gross_adjusted
mean_aggregates = {}
for feature in categorical_features:
    mean_feature_name = f"{feature}_mean_gross"
    data[mean_feature_name] = data.groupby(feature)[target].transform('mean')
    correlation = data[[mean_feature_name, target]].corr().iloc[0, 1]
    mean_aggregates[mean_feature_name] = correlation

mean_aggregates_df = pd.DataFrame.from_dict(mean_aggregates, orient='index', columns=['Correlation']).sort_values(by='Correlation', ascending=False)

plt.figure(figsize=(6, 8))
sns.heatmap(mean_aggregates_df, annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
plt.title('Correlation of Mean Aggregations with Adjusted Gross')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "mean_aggregations_correlation_heatmap.png"))
plt.close()

# Step 5: Save suggestions to file
with open(suggestions_file, 'w') as file:
    file.write("All Ratios and Interactions with Correlations:\n")
    for feature, corr in sorted(all_features, key=lambda x: abs(x[1]), reverse=True):
        file.write(f"- {feature}: correlation with {target} = {corr:.2f}\n")
    
    file.write("\nSuggested Features for Prediction (correlation > 0.60):\n")
    for feature, corr in sorted(suggested_features, key=lambda x: abs(x[1]), reverse=True):
        file.write(f"- {feature}: correlation with {target} = {corr:.2f}\n")
    
    file.write("\nSuggested Mean Aggregations for Categorical Variables (correlation > 0.60):\n")
    for feature, corr in sorted(mean_aggregates.items(), key=lambda x: abs(x[1]), reverse=True):
        if abs(corr) > 0.60:
            file.write(f"- {feature}: correlation with {target} = {corr:.2f}\n")

print(f"All correlations and suggestions saved to {suggestions_file}")