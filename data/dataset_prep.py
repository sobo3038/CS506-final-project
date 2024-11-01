import pandas as pd

# Load the adjusted dataset
data = pd.read_csv('data/movies_adjusted.csv')

# Ensure we only work with adjusted monetary values and drop non-predictive columns
data = data.drop(columns=['gross', 'budget', 'score', 'votes'], errors='ignore')

# Extract `release_month` from the `released` column
data['released'] = pd.to_datetime(data['released'], errors='coerce')
data['release_month'] = data['released'].dt.month

# Create new features
data['budget_adjusted_times_runtime'] = data['budget_adjusted'] * data['runtime']
data['budget_adjusted_times_year'] = data['budget_adjusted'] * data['year']
data['budget_adjusted_to_year_ratio'] = data['budget_adjusted'] / data['year']

# Create mean aggregation features
data['writer_mean_gross'] = data.groupby('writer')['gross_adjusted'].transform('mean')
data['director_mean_gross'] = data.groupby('director')['gross_adjusted'].transform('mean')

# Save the final dataset as a new CSV
data.to_csv('data/final_prediction_dataset.csv', index=False)

print("Final prediction dataset created and saved as 'final_prediction_dataset.csv'.")
