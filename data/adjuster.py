import pandas as pd

# Load the CPI data
cpi_data = pd.DataFrame({
    'Year': [1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 
             1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
             2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'CPI': [82.7, 90.6, 97.0, 99.6, 104.9, 108.5, 110.1, 114.9, 119.7, 125.3, 132.6, 137.2, 141.4, 
            145.3, 149.3, 153.2, 157.9, 161.2, 163.7, 167.8, 173.6, 177.5, 180.9, 184.6, 190.2, 197.4, 
            202.6, 208.9, 216.2, 215.9, 218.6, 224.3, 229.6, 233.5, 237.1, 237.8, 241.2, 246.2, 252.1, 
            257.6, 260.2, 275.7, 296.9, 307.7, 314.3]
})

# Set the base year for adjustment (e.g., 2020)
base_year = 2020
base_cpi = cpi_data.loc[cpi_data['Year'] == base_year, 'CPI'].values[0]

# Load the movie dataset
movies_data = pd.read_csv('data/movies.csv')

# Merge CPI data with movie data on the 'year' column
movies_data = movies_data.merge(cpi_data, left_on='year', right_on='Year', how='left')

# Calculate the adjustment factor
movies_data['adjustment_factor'] = base_cpi / movies_data['CPI']

# Adjust 'gross' and 'budget' columns to 2020 dollars
movies_data['gross_adjusted'] = movies_data['gross'] * movies_data['adjustment_factor']
movies_data['budget_adjusted'] = movies_data['budget'] * movies_data['adjustment_factor']

# Drop the original non-adjusted money columns and other unnecessary columns
movies_data = movies_data.drop(columns=['Year', 'CPI', 'adjustment_factor', 'gross', 'budget'])

# Save the adjusted dataset as a new CSV
movies_data.to_csv('data/movies_adjusted.csv', index=False)

print("Inflation-adjusted dataset created and saved as 'movies_adjusted.csv'")