import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Load data (assuming the file is named 'dataset.csv')
data = pd.read_csv('data/movies.csv')

# Ensure the folder exists
output_folder = "data_engineering_images"
os.makedirs(output_folder, exist_ok=True)

# Extract date only from 'released' and parse it
data['released'] = data['released'].str.extract(r'(\w+ \d{1,2}, \d{4})')[0]
data['released'] = pd.to_datetime(data['released'], format='%B %d, %Y', errors='coerce')
data['release_year'] = data['released'].dt.year
data['release_month'] = data['released'].dt.month

# Function to save plots
def save_plot(fig, filename):
    fig.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    plt.close(fig)

# 1. Histograms for Numerical Variables with Binned Averages for budget, gross, and votes
for column in ['votes', 'budget', 'gross']:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Frequency histogram
    axes[0].hist(data[column].dropna(), bins=30)
    axes[0].set_title(f'Frequency of {column}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    
    # Binned average plot
    if data[column].notna().any():
        # Define bins for each variable
        bins = np.linspace(data[column].min(), data[column].max(), 10)  # Adjust bin count if necessary
        data[f'{column}_binned'] = pd.cut(data[column], bins=bins)
        binned_avg = data.groupby(f'{column}_binned')['score'].mean()
        
        # Plotting binned averages
        binned_avg.plot(kind='bar', ax=axes[1])
        axes[1].set_title(f'Average Score by Binned {column.capitalize()}')
        axes[1].set_xlabel(f'{column.capitalize()} Range')
        axes[1].set_ylabel('Average Score')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, 'No data available', ha='center')
    
    fig.suptitle(f'Analysis of {column.capitalize()}')
    fig.tight_layout()
    save_plot(fig, f'analysis_{column}_binned.png')

# 2. Scatter plots with `score`
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, column in zip(axes.flatten(), ['votes', 'budget', 'gross', 'runtime']):
    ax.scatter(data[column], data['score'], alpha=0.5)
    ax.set_title(f'Score vs. {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Score')
fig.suptitle('Scatter Plots of Score with Other Variables')
fig.tight_layout()
save_plot(fig, 'scatter_score_variables.png')

# 3. Box plots for categorical variables with 90° rotation
for column in ['genre', 'rating', 'country']:
    fig, ax = plt.subplots(figsize=(10, 6))
    data.boxplot(column='score', by=column, ax=ax)
    ax.set_title(f'Score by {column.capitalize()}')
    ax.set_xlabel(column.capitalize())
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=90)
    plt.suptitle('')
    fig.tight_layout()
    save_plot(fig, f'boxplot_score_{column}.png')

# 4. Average score over time (by year and by month)
fig, ax = plt.subplots()
average_score_by_year = data.groupby('release_year')['score'].mean()
ax.plot(average_score_by_year.index, average_score_by_year.values)
ax.set_title('Average Score by Year')
ax.set_xlabel('Year')
ax.set_ylabel('Average Score')
fig.tight_layout()
save_plot(fig, 'average_score_by_year.png')

fig, ax = plt.subplots()
average_score_by_month = data.groupby('release_month')['score'].mean()
ax.plot(average_score_by_month.index, average_score_by_month.values)
ax.set_title('Average Score by Month')
ax.set_xlabel('Month')
ax.set_ylabel('Average Score')
fig.tight_layout()
save_plot(fig, 'average_score_by_month.png')

# 5. Heatmap for average score by year and month
average_score_by_year_month = data.groupby(['release_year', 'release_month'])['score'].mean().unstack()
if not average_score_by_year_month.empty:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(average_score_by_year_month, cmap="YlGnBu", ax=ax)
    ax.set_title('Average Score by Year and Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    fig.tight_layout()
    save_plot(fig, 'heatmap_score_by_year_month.png')
else:
    print("No data available for year-month heatmap.")

# 6. Top 10 by Directors, Writers, Stars, and Companies
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for ax, column, label in zip(axes.flatten(), ['director', 'writer', 'star', 'company'], ['Directors', 'Writers', 'Stars', 'Companies']):
    top_10 = data.groupby(column)['score'].mean().nlargest(10)
    top_10.plot(kind='bar', ax=ax)
    ax.set_title(f'Top 10 {label} by Average Score')
    ax.set_xlabel(label)
    ax.set_ylabel('Average Score')
    ax.tick_params(axis='x', rotation=90)
fig.suptitle('Top Contributors by Average Score')
fig.tight_layout()
save_plot(fig, 'top_contributors_average_score.png')

# 7. Enhanced Correlation Analysis with Pair Plot
fig = sns.pairplot(data[['score', 'votes', 'budget', 'gross', 'runtime']].dropna())
fig.fig.suptitle('Pair Plot of Numerical Variables', y=1.02)
fig.savefig(os.path.join(output_folder, 'enhanced_pair_plot.png'), bbox_inches='tight')

print("Graphs have been generated and saved in the 'data_engineering_images' folder.")