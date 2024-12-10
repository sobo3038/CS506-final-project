import sys
import pandas as pd
import xgboost as xgb
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Dynamically add the directory containing xgboost_model.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../boost_models')))

from xgboost_model import model_pipeline  # Import the model pipeline


# Load the pre-trained XGBoost model
model_path = os.path.join(os.path.dirname(__file__), '../webpage/xgboost_model.json')
model = xgb.XGBRegressor()
model.load_model(model_path)

# Function to calculate the CPI adjustment factor
# adjustment_factor_2020 = CPI_2020 / CPI_year
# 2020 USD = release_year_USD * adjustment_factor_2020
def get_inflation_adjustment_factor(cpi_data, year, base_year=2020):
    try:
        cpi_base_year = cpi_data.loc[cpi_data['Year'] == base_year, 'CPI'].values[0]
        cpi_year = cpi_data.loc[cpi_data['Year'] == year, 'CPI'].values[0]
        adjustment_factor = cpi_base_year / cpi_year
        return adjustment_factor
    except:
        # If CPI data is not available, return factor of 1 (no adjustment)
        return 1

# Function to calculate confidence intervals
# Here, a simplistic +/-10% margin is used as a placeholder.
def calculate_confidence_interval(prediction):
    if np.isfinite(prediction):
        margin_of_error = 0.1 * prediction
        lower_bound = prediction - margin_of_error
        upper_bound = prediction + margin_of_error
        return lower_bound, upper_bound
    else:
        return None, None

# Function to generate explanation based on inputs
def generate_explanation(budget, runtime, release_month, company, writer, director):
    explanation = "The prediction is influenced by "
    if budget > 50_000_000:
        explanation += "a high budget, "
    elif budget < 10_000_000:
        explanation += "a low budget, "
    if runtime > 120:
        explanation += "a long runtime, "
    elif runtime < 90:
        explanation += "a short runtime, "
    if release_month in [6, 7, 12]:
        explanation += "and a high-demand release month."
    else:
        explanation += "and a standard release month."

    explanation += f" The company {company} and the creative team (writer {writer} and director {director}) also play a significant role."
    return explanation

# Function to display feature importance
def display_feature_importance(model, feature_names):
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)


    print(f"\n--- Feature Importances: ---")
    print(importance_df.to_string(index=False))


# Function to perform prediction for given inputs
def predict_revenue(input_data, cpi_data):
    # Predict revenue (assuming no log transform was used during training)
    prediction_raw = model.predict(input_data)[0]

    # If the model was trained on logged targets, uncomment the following line:
    # prediction_raw = np.exp(prediction_raw)

    # `prediction_raw` is now in release-year USD since that's how we assumed we trained it.

    # Get the inflation adjustment factor to convert from release-year to 2020 USD
    input_year = input_data['budget_adjusted_times_year'].values[0] / input_data['budget_adjusted'].values[0]
    adjustment_factor_2020 = get_inflation_adjustment_factor(cpi_data, int(input_year))

    # Convert to 2020 USD
    prediction_2020 = prediction_raw * adjustment_factor_2020

    # Calculate confidence intervals
    lower_bound, upper_bound = calculate_confidence_interval(prediction_raw)

    return prediction_raw, prediction_2020, lower_bound, upper_bound

# Main function to perform predictions and evaluate model
def main():
    # Load the CPI data
    cpi_data = pd.DataFrame({
        'Year': [1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
                 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                 2020, 2021, 2022, 2023, 2024],
        'CPI': [82.7, 90.6, 97.0, 99.6, 104.9, 108.5, 110.1, 114.9, 119.7, 125.3,
                132.6, 137.2, 141.4, 145.3, 149.3, 153.2, 157.9, 161.2, 163.7, 167.8,
                173.6, 177.5, 180.9, 184.6, 190.2, 197.4, 202.6, 208.9, 216.2, 215.9,
                218.6, 224.3, 229.6, 233.5, 237.1, 237.8, 241.2, 246.2, 252.1, 257.6,
                260.2, 275.7, 296.9, 307.7, 314.3]
    })

    # Example Test Cases
    test_cases = [
        {
            "movie": "Guardians of the Galaxy Vol. 2",
            "year": 2017,
            "release_month": 5,  # May
            "budget": 200_000_000,  # USD
            "company": "Marvel Studios",
            "runtime": 136.0,  # minutes
            "writer": "James Gunn",
            "director": "James Gunn",
            "actual_revenue": 773_328_629  # Actual revenue for validation (replace with actual value if different)
        },
        # You can add more test cases here
    ]

    predictions = []
    actuals = []

    for case in test_cases:
        print(f"\n--- Prediction for '{case['movie']}' ---")
        input_year = case['year']
        input_release_month = case['release_month']
        input_budget = case['budget']
        input_company = case['company']
        input_runtime = case['runtime']
        input_writer = case['writer']
        input_director = case['director']
        actual_revenue = case.get('actual_revenue', None)  # Optional: provide actual revenue for RMSE and R²

        # Prepare the input data
        input_data = pd.DataFrame({
            "budget_adjusted": [input_budget],
            "runtime": [input_runtime],
            "release_month": [input_release_month],
            "budget_adjusted_times_runtime": [input_budget * input_runtime],
            "budget_adjusted_times_year": [input_budget * input_year]
        })

        # Display input data for debugging
        print("Input Data for Prediction:")
        print(input_data)

        # Perform prediction
        prediction_raw, prediction_2020, lower_bound, upper_bound = predict_revenue(input_data, cpi_data)

        # Generate explanation
        explanation = generate_explanation(
            budget=input_budget,
            runtime=input_runtime,
            release_month=input_release_month,
            company=input_company,
            writer=input_writer,
            director=input_director
        )

        # Display results
        print(f"\n--- Prediction Results: ---")
        print(f"Predicted Revenue         (2017 USD): ${prediction_raw:,.2f}" if np.isfinite(prediction_raw) else "Predicted Revenue: Unavailable")
        print(f"Predicted Revenue         (2020 USD): ${prediction_2020:,.2f}" if np.isfinite(prediction_2020) else "Predicted Revenue (2020 USD): Unavailable")
        if lower_bound is not None and upper_bound is not None:
            print(f"\nConfidence Interval: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
        else:
            print("Confidence Interval: Unavailable")

        print("Acutal Revenue: $863,756,051")

        # Collect predictions and actuals for RMSE and R²
        if actual_revenue is not None and np.isfinite(prediction_raw):
            predictions.append(prediction_raw)
            actuals.append(actual_revenue)

    # Display feature importance using the first test case's features
    if test_cases:
        feature_names = ["budget_adjusted", "runtime", "release_month", "budget_adjusted_times_runtime", "budget_adjusted_times_year"]
        display_feature_importance(model, feature_names)

if __name__ == "__main__":
    main()
