from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)

# Load the pre-trained XGBoost model
model = xgb.XGBRegressor()
model.load_model("xgboost_model.json")

# Set the Matplotlib backend to 'Agg' to avoid GUI issues on macOS
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

# Function to find similar movies
def get_similar_movies(budget, year):
    data = pd.read_csv('../data/final_prediction_dataset.csv')
    similar_movies = data[(data['budget_adjusted'] > budget * 0.8) & 
                          (data['budget_adjusted'] < budget * 1.2) & 
                          (data['year'] == year)]
    return [{"name": row['name'], "gross_adjusted": f"${row['gross_adjusted']:,.2f}"} 
            for _, row in similar_movies.head(3).iterrows()]

# Function to calculate confidence intervals
def calculate_confidence_interval(prediction):
    if np.isfinite(prediction):
        margin_of_error = 0.1 * prediction
        lower_bound = prediction - margin_of_error
        upper_bound = prediction + margin_of_error
        return f"${lower_bound:,.2f}", f"${upper_bound:,.2f}"
    else:
        return "Unavailable", "Unavailable"

# Function to generate explanation based on inputs
def generate_explanation(budget, runtime, release_month):
    explanation = "The prediction is influenced by "
    if budget > 50000000:
        explanation += "a high budget, "
    elif budget < 10000000:
        explanation += "a low budget, "
    if runtime > 120:
        explanation += "a long runtime, "
    elif runtime < 90:
        explanation += "a short runtime, "
    if release_month in [6, 7, 12]:
        explanation += "and a high-demand release month."
    else:
        explanation += "and a standard release month."
    return explanation

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    lower_bound, upper_bound = None, None
    explanation = ""
    similar_movies = []
    feature_importance_img = get_feature_importance()
    
    if request.method == "POST":
        year = int(request.form.get("year"))
        budget = float(request.form.get("budget"))
        runtime = float(request.form.get("runtime"))
        release_month = int(request.form.get("release_month"))

        input_data = pd.DataFrame({
            "budget_adjusted": [budget],
            "runtime": [runtime],
            "release_month": [release_month],
            "budget_adjusted_times_runtime": [budget * runtime],
            "budget_adjusted_times_year": [budget * year]
        })

        # Debug: Print input data and prediction log
        print("Input Data for Prediction:", input_data)
        
        # Prediction without log transformation (alternative approach)
        prediction_raw = model.predict(input_data)[0]
        prediction = f"${prediction_raw:,.2f}" if np.isfinite(prediction_raw) else "Unavailable"
        
        lower_bound, upper_bound = calculate_confidence_interval(prediction_raw)
        explanation = generate_explanation(budget, runtime, release_month)
        similar_movies = get_similar_movies(budget, year)

    return render_template("index.html", 
                           prediction=prediction, 
                           lower_bound=lower_bound, 
                           upper_bound=upper_bound, 
                           explanation=explanation, 
                           similar_movies=similar_movies, 
                           feature_importance_img=feature_importance_img)

if __name__ == "__main__":
    app.run(debug=True)
