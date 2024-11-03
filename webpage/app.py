from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Load the pre-trained XGBoost model
model = xgb.XGBRegressor()
model.load_model("xgboost_model.json")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get user input from the form
        year = int(request.form.get("year"))
        budget = float(request.form.get("budget"))
        runtime = float(request.form.get("runtime"))
        release_month = int(request.form.get("release_month"))

        # Prepare input data for the model
        input_data = pd.DataFrame({
            "budget_adjusted": [budget],
            "runtime": [runtime],
            "release_month": [release_month],
            "budget_adjusted_times_runtime": [budget * runtime],
            "budget_adjusted_times_year": [budget * year]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

    # Render index.html and pass the prediction to the template
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
