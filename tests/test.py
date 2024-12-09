import sys
import os
import unittest
import pandas as pd
import numpy as np

# Dynamically add the directory containing xgboost_model.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../boost_models')))

from xgboost_model import model_pipeline  # Import the model pipeline

def get_inflation_adjustment_factor(cpi_data, year, base_year=2020):
    """
    Returns the inflation adjustment factor to convert values from `year` to `base_year`.
    """
    try:
        cpi_base_year = cpi_data.loc[cpi_data['Year'] == base_year, 'CPI'].values[0]
        cpi_year = cpi_data.loc[cpi_data['Year'] == year, 'CPI'].values[0]
        adjustment_factor = cpi_base_year / cpi_year
        return adjustment_factor
    except Exception as e:
        print(f"Error in inflation adjustment factor calculation: {e}")
        return 1  # Default factor if CPI data is not available

class TestXGBoostModel(unittest.TestCase):
    def test_revenue_prediction(self):
        """
        Test the model's prediction for Guardians of the Galaxy Vol. 2.
        """
        year = 2017
        release_month = 5  # May
        budget = 200_000_000  # USD
        runtime = 136  # Minutes

        # Prepare input data for the model with all expected columns
        input_data = pd.DataFrame({
            "name": ["Guardians of the Galaxy Vol. 2"],
            "star": ["Chris Pratt"],
            "genre": ["Action"],
            "company": ["Marvel Studios"],
            "rating": ["PG-13"],
            "writer_mean_gross": [100_000_000],  # Placeholder value
            "director": ["James Gunn"],
            "country": ["United States"],
            "year": [year],
            "writer": ["James Gunn"],
            "director_mean_gross": [500_000_000],  # Placeholder value
            "budget_adjusted": [budget],
            "runtime": [runtime],
            "release_month": [release_month],
            "budget_adjusted_times_runtime": [budget * runtime],
            "budget_adjusted_times_year": [budget * year]
        })

        # Predict revenue
        prediction_log = model_pipeline.predict(input_data)[0]  # Log space prediction
        prediction_raw = np.exp(prediction_log)  # Convert back to original scale

        # Adjust to 2020 USD using CPI data
        cpi_data = pd.DataFrame({
            'Year': [2017, 2020],
            'CPI': [241.2, 257.6]  # Example CPI values for testing
        })
        adjustment_factor_2020 = get_inflation_adjustment_factor(cpi_data, year)
        prediction_2020 = prediction_raw * adjustment_factor_2020

        # Output for comparison
        print("\nPredicted revenue:")
        print(f"Predicted Revenue (2017 USD): ${prediction_raw:,.2f}")
        print(f"Predicted Revenue (2020 USD): ${prediction_2020:,.2f}")

        # Assertions (adjust thresholds as needed)
        self.assertGreater(prediction_raw, 800_000_000, "Prediction in 2017 USD seems too low.")
        self.assertGreater(prediction_2020, 850_000_000, "Prediction in 2020 USD seems too low.")

if __name__ == "__main__":
    unittest.main()