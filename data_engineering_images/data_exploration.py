import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Load data (assuming the file is named 'data/movies_adjusted.csv')
data = pd.read_csv('data/movies_adjusted.csv')

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

# 1. Histograms for Numerical Variables with Binned Averages for adjusted budget, votes, and runtime
for column in ['votes', 'budget_adjusted', 'runtime']:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Frequency histogram
    axes[0].hist(data[column].dropna(), bins=30)
    axes[0].set_title(f'Frequency of {column}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    
    # Binned average plot with adjusted gross as target
    if data[column].notna().any():
        # Define bins for each variable
        bins = np.linspace(data[column].min(), data[column].max(), 10)
        data[f'{column}_binned'] = pd.cut(data[column], bins=bins)
        binned_avg = data.groupby(f'{column}_binned')['gross_adjusted'].mean()
        
        # Plotting binned averages
        binned_avg.plot(kind='bar', ax=axes[1])
        axes[1].set_title(f'Average Adjusted Gross by Binned {column.capitalize()}')
        axes[1].set_xlabel(f'{column.capitalize()} Range')
        axes[1].set_ylabel('Average Adjusted Gross')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, 'No data available', ha='center')
    
    fig.suptitle(f'Analysis of {column.capitalize()} (Adjusted)')
    fig.tight_layout()
    save_plot(fig, f'analysis_{column}_binned_adjusted.png')

# 2. Scatter plots with `gross_adjusted`
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, column in zip(axes.flatten(), ['votes', 'budget_adjusted', 'runtime', 'release_year']):
    ax.scatter(data[column], data['gross_adjusted'], alpha=0.5)
    ax.set_title(f'Adjusted Gross vs. {column.capitalize()}')
    ax.set_xlabel(column.capitalize())
    ax.set_ylabel('Adjusted Gross')
fig.suptitle('Scatter Plots of Adjusted Gross with Other Variables')
fig.tight_layout()
save_plot(fig, 'scatter_gross_adjusted_variables.png')

# 3. Box plots for categorical variables with 90° rotation
for column in ['genre', 'rating', 'country']:
    fig, ax = plt.subplots(figsize=(10, 6))
    data.boxplot(column='gross_adjusted', by=column, ax=ax)
    ax.set_title(f'Adjusted Gross by {column.capitalize()}')
    ax.set_xlabel(column.capitalize())
    ax.set_ylabel('Adjusted Gross')
    ax.tick_params(axis='x', rotation=90)
    plt.suptitle('')
    fig.tight_layout()
    save_plot(fig, f'boxplot_gross_adjusted_{column}.png')

# 4. Average adjusted gross over time (by year and by month)
fig, ax = plt.subplots()
average_gross_by_year = data.groupby('release_year')['gross_adjusted'].mean()
ax.plot(average_gross_by_year.index, average_gross_by_year.values)
ax.set_title('Average Adjusted Gross by Year')
ax.set_xlabel('Year')
ax.set_ylabel('Average Adjusted Gross')
fig.tight_layout()
save_plot(fig, 'average_gross_adjusted_by_year.png')

fig, ax = plt.subplots()
average_gross_by_month = data.groupby('release_month')['gross_adjusted'].mean()
ax.plot(average_gross_by_month.index, average_gross_by_month.values)
ax.set_title('Average Adjusted Gross by Month')
ax.set_xlabel('Month')
ax.set_ylabel('Average Adjusted Gross')
fig.tight_layout()
save_plot(fig, 'average_gross_adjusted_by_month.png')

# 5. Heatmap for average adjusted gross by year and month
average_gross_by_year_month = data.groupby(['release_year', 'release_month'])['gross_adjusted'].mean().unstack()
if not average_gross_by_year_month.empty:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(average_gross_by_year_month, cmap="YlGnBu", ax=ax)
    ax.set_title('Average Adjusted Gross by Year and Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    fig.tight_layout()
    save_plot(fig, 'heatmap_gross_adjusted_by_year_month.png')
else:
    print("No data available for year-month heatmap.")

# 6. Top 10 by Directors, Writers, Stars, and Companies for Adjusted Gross
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for ax, column, label in zip(axes.flatten(), ['director', 'writer', 'star', 'company'], ['Directors', 'Writers', 'Stars', 'Companies']):
    top_10 = data.groupby(column)['gross_adjusted'].mean().nlargest(10)
    top_10.plot(kind='bar', ax=ax)
    ax.set_title(f'Top 10 {label} by Average Adjusted Gross')
    ax.set_xlabel(label)
    ax.set_ylabel('Average Adjusted Gross')
    ax.tick_params(axis='x', rotation=90)
fig.suptitle('Top Contributors by Average Adjusted Gross')
fig.tight_layout()
save_plot(fig, 'top_contributors_average_gross_adjusted.png')

# 7. Enhanced Correlation Analysis with Pair Plot for Adjusted Gross including Score
fig = sns.pairplot(data[['gross_adjusted', 'votes', 'budget_adjusted', 'runtime', 'release_year', 'score']].dropna())
fig.fig.suptitle('Pair Plot of Numerical Variables (with Adjusted Gross and Score)', y=1.02)
fig.savefig(os.path.join(output_folder, 'enhanced_pair_plot_gross_adjusted_with_score.png'), bbox_inches='tight')

# 8. Additional Analysis: Relationship between Score and Adjusted Gross
# Scatter plot of Score vs. Adjusted Gross
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data['score'], data['gross_adjusted'], alpha=0.5)
ax.set_title('Score vs. Adjusted Gross')
ax.set_xlabel('Score')
ax.set_ylabel('Adjusted Gross')
fig.tight_layout()
save_plot(fig, 'scatter_score_vs_adjusted_gross.png')

# Binned Score vs. Average Adjusted Gross
fig, ax = plt.subplots(figsize=(10, 6))
score_bins = pd.cut(data['score'], bins=np.arange(0, 10.5, 1))  # Bins from 0 to 10 with a step of 1
average_gross_by_score_bin = data.groupby(score_bins)['gross_adjusted'].mean()
average_gross_by_score_bin.plot(kind='bar', ax=ax)
ax.set_title('Average Adjusted Gross by Binned Score')
ax.set_xlabel('Score Range')
ax.set_ylabel('Average Adjusted Gross')
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
save_plot(fig, 'binned_score_vs_average_adjusted_gross.png')

print("Graphs have been generated and saved in the 'data_engineering_images' folder.")