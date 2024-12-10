from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import base64
import os
import numpy as np


app = Flask(__name__)


# Load the pre-trained XGBoost model
model = xgb.XGBRegressor()
model_path = os.path.join(os.path.dirname(__file__), "xgboost_model.json")
model.load_model(model_path)


# Set the Matplotlib backend to 'Agg' to avoid GUI issues on some systems
plt.switch_backend("Agg")


# Function to get feature importance as an image
def get_feature_importance():
   feature_importance = model.feature_importances_
   features = ["budget_adjusted", "runtime", "release_month", "budget_adjusted_times_runtime", "budget_adjusted_times_year"]


   # Plotting the feature importances
   plt.figure(figsize=(6, 4))
   plt.barh(features, feature_importance, color="skyblue")
   plt.xlabel("Importance")
   plt.title("Feature Importance for Revenue Prediction")
   plt.tight_layout()


   # Save plot to a string buffer
   buf = io.BytesIO()
   plt.savefig(buf, format="png")
   buf.seek(0)
   img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
   buf.close()
   return img_str


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
       return f"${lower_bound:,.2f}", f"${upper_bound:,.2f}"
   else:
       return "Unavailable", "Unavailable"


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


@app.route("/", methods=["GET", "POST"])
def index():
   prediction = None
   prediction_inflation_2020 = ""
   prediction_inflation_release_year = ""
   lower_bound, upper_bound = None, None
   explanation = ""
   feature_importance_img = get_feature_importance()


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


   year = None  # Initialize `year` in case of GET request


   if request.method == "POST":
       year = int(request.form.get("year"))
       budget = float(request.form.get("budget"))
       runtime = float(request.form.get("runtime"))
       release_month = int(request.form.get("release_month"))
       company = request.form.get("company")
       writer = request.form.get("writer")
       director = request.form.get("director")


       # Prepare the input data
       input_data = pd.DataFrame({
           "budget_adjusted": [budget],
           "runtime": [runtime],
           "release_month": [release_month],
           "budget_adjusted_times_runtime": [budget * runtime],
           "budget_adjusted_times_year": [budget * year]
       })


       # DEBUG: Print input data
       print("Input Data for Prediction:", input_data)


       # Predict revenue (assuming no log transform was used during training)
       prediction_raw = model.predict(input_data)[0]


       # If the model was trained on logged targets, uncomment the following line:
       # prediction_raw = np.exp(prediction_raw)


       # `prediction_raw` is now in release-year USD since that's how we assumed we trained it.


       # Get the inflation adjustment factor to convert from release-year to 2020 USD
       adjustment_factor_2020 = get_inflation_adjustment_factor(cpi_data, year)


       # Convert to 2020 USD
       prediction_2020 = prediction_raw * adjustment_factor_2020


       # Format outputs
       prediction_inflation_release_year = f"${prediction_raw:,.2f}" if np.isfinite(prediction_raw) else "Unavailable"
       prediction_inflation_2020 = f"${prediction_2020:,.2f}" if np.isfinite(prediction_2020) else "Unavailable"


       # Assign for display in the template
       prediction = prediction_inflation_release_year


       lower_bound, upper_bound = calculate_confidence_interval(prediction_raw)
       explanation = generate_explanation(budget, runtime, release_month, company, writer, director)


   return render_template("index.html",
                          prediction=prediction,
                          prediction_inflation_2020=prediction_inflation_2020,
                          prediction_inflation_release_year=prediction_inflation_release_year,
                          year=year,
                          lower_bound=lower_bound,
                          upper_bound=upper_bound,
                          explanation=explanation,
                          feature_importance_img=feature_importance_img)


if __name__ == "__main__":
   app.run(debug=True)



