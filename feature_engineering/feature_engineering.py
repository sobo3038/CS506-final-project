import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Load the adjusted dataset
data = pd.read_csv('data/movies_adjusted.csv')

# Ensure we only work with adjusted monetary values
data = data.drop(columns=['gross', 'budget'], errors='ignore')

# Ensure target variable is correctly specified
target = 'gross_adjusted'
output_folder = "feature_engineering"
os.makedirs(output_folder, exist_ok=True)
suggestions_file = os.path.join(output_folder, 'suggestions.txt')

# List to store all calculated features and their correlations
all_features = []
suggested_features = []

# Step 1: Original Feature Correlations with gross_adjusted
# Filter for numerical columns only
numerical_data = data.select_dtypes(include=[np.number])
original_corr = numerical_data.corr()[[target]].drop(target).sort_values(by=target, ascending=False)

# Step 2: Calculate Ratios and Interactions, save correlations
def calculate_feature_ratios_interactions(data, target='gross_adjusted', suggestion_threshold=0.51):
    numerical_features = data.select_dtypes(include=[np.number]).columns.drop(target)
    
    for (feature1, feature2) in combinations(numerical_features, 2):
        # Ratios
        if data[feature2].abs().min() > 1e-9:  # Avoid division by zero
            ratio_feature_name = f"{feature1}_to_{feature2}_ratio"
            data[ratio_feature_name] = data[feature1] / data[feature2]
            correlation = data[[target, ratio_feature_name]].corr().iloc[0, 1]
            all_features.append((ratio_feature_name, correlation))
            if abs(correlation) > suggestion_threshold:
                suggested_features.append((ratio_feature_name, correlation))
            data.drop(columns=[ratio_feature_name], inplace=True)  # Clean up temporary column

        # Interactions
        interaction_feature_name = f"{feature1}_times_{feature2}"
        data[interaction_feature_name] = data[feature1] * data[feature2]
        correlation = data[[target, interaction_feature_name]].corr().iloc[0, 1]
        all_features.append((interaction_feature_name, correlation))
        if abs(correlation) > suggestion_threshold:
            suggested_features.append((interaction_feature_name, correlation))
        data.drop(columns=[interaction_feature_name], inplace=True)  # Clean up temporary column

# Run feature calculation and suggestion generation
calculate_feature_ratios_interactions(data, target)

# Step 3: Save all feature correlations and suggestions to file
with open(suggestions_file, 'w') as file:
    file.write("All Ratios and Interactions with Correlations:\n")
    for feature, corr in sorted(all_features, key=lambda x: abs(x[1]), reverse=True):
        file.write(f"- {feature}: correlation with {target} = {corr:.2f}\n")
    file.write("\nSuggested Features for Prediction (correlation > 0.75):\n")
    for feature, corr in sorted(suggested_features, key=lambda x: abs(x[1]), reverse=True):
        file.write(f"- {feature}: correlation with {target} = {corr:.2f}\n")

print(f"All correlations and suggestions saved to {suggestions_file}")

# Step 4: Generate Heatmap for Original Feature Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(original_corr, annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
plt.title('Correlation of Original Features with Adjusted Gross')
plt.tight_layout()

# Save heatmap
original_corr_plot_path = os.path.join("feature_engineering", "original_features_correlation_heatmap.png")
plt.savefig(original_corr_plot_path)
plt.close()
print(f"Original feature correlation heatmap saved to {original_corr_plot_path}")

# Step 5: Generate Heatmap for All Ratios and Interactions
# Convert all_features to DataFrame for heatmap
all_features_df = pd.DataFrame(all_features, columns=['Feature', 'Correlation']).set_index('Feature')

# Plot the heatmap
plt.figure(figsize=(10, len(all_features_df) // 4))  # Adjust height based on the number of features
sns.heatmap(all_features_df.T, annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
plt.title('Correlation of Ratios and Interactions with Adjusted Gross')
plt.tight_layout()

# Save heatmap
interaction_corr_plot_path = os.path.join("feature_engineering", "ratios_interactions_correlation_heatmap.png")
plt.savefig(interaction_corr_plot_path)
plt.close()
print(f"Ratios and interactions correlation heatmap saved to {interaction_corr_plot_path}")