# CS506 Final Project: Box Office Revenue Prediction

**Description:**
This project will develop a predictive model that can forecast the box office success of upcoming movies based on various factors (cast, director, budget, genre, release date, etc.). The model will provide insights into the key elements that contribute to a movie’s box office performance. The data will be collected from **The Movie Database (TMDb) API**.

**Clear Goal:**
- Predict the box office revenue for an upcoming movie before its release using historical data on similar movies.
- Analyze the importance of different features (e.g., cast, budget, genre, etc.) in determining box office success.

## Data Collection

**Data Sources:**
- **The Movie Database (TMDb) API**: Provides movie metadata such as budget, genre, cast, and director.
- **Kaggle Datasets**: [Movies Dataset](https://www.kaggle.com/datasets/danielgrijalvas/movies)

**Features to Collect:**
- Cast
- Director
- Budget
- Release Date
- Genre

## Data Modeling

For the box office revenue predictor, the goal is to predict a movie's box office revenue based on features such as cast, director, budget, release date, and genre. After defining the target variable (revenue), we will test several models:
- Start with **Linear Regression** as a baseline.
- Progress to more advanced models such as **Decision Trees** or **Random Forest**.
- Evaluate model performance using **RMSE** and **R²** scores.

## Data Visualization

**Model Visualizations:**
- **Predicted vs. Actual Revenue**: A scatter plot to visualize model accuracy.
- **Feature Importance**: A bar chart to show which factors most influence box office revenue.
- **User Interface**: Allows for users to input details of a movie and predict its potential box office revenue.

**Data Analysis:**
- Scatter plots comparing features like budget and revenue.
- Bar charts of feature correlations.
- Time-based visualizations showing revenue trends based on release date or season.

## Test Plan

- Split the dataset into training and testing sets (e.g., 80-20 or 70-30) to evaluate model performance on unseen data.
- Model performance will be evaluated using metrics such as **RMSE**.
- The model will also be tested on movies released after training data ends to evaluate how well it generalizes to unseen data.
