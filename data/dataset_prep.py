import pandas as pd

# Load the adjusted dataset
data = pd.read_csv('data/movies_adjusted.csv')

# Extract date only from 'released' and parse it
data['released'] = data['released'].str.extract(r'(\w+ \d{1,2}, \d{4})')[0]
data['released'] = pd.to_datetime(data['released'], format='%B %d, %Y', errors='coerce')
data['release_month'] = data['released'].dt.month

# Ensure we only work with adjusted monetary values and drop non-predictive columns
data = data.drop(columns=['gross', 'budget', 'score', 'votes', 'released'], errors='ignore')

# Create new features
data['budget_adjusted_times_runtime'] = data['budget_adjusted'] * data['runtime']
data['budget_adjusted_times_year'] = data['budget_adjusted'] * data['year']

# Create mean aggregation features
data['writer_mean_gross'] = data.groupby('writer')['gross_adjusted'].transform('mean')
data['director_mean_gross'] = data.groupby('director')['gross_adjusted'].transform('mean')

# Save the final dataset as a new CSV
data.to_csv('data/final_prediction_dataset.csv', index=False)

print("Final prediction dataset created and saved as 'final_prediction_dataset.csv'.")
