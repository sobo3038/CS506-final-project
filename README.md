# CS506 Final Project: Box Office Revenue Prediction

# Midterm Report:
--------------
## Youtube Link:
https://youtu.be/Uk0iqkvShx0

## Step 1: Data Preparation and Inflation Adjustment
To create a reliable model, we began by standardizing financial data over time, as movies in the dataset span multiple decades. Since inflation affects both the meaning and value of financial figures like budgets and revenue, we couldn’t directly compare these amounts across different years without adjustment. We used the Consumer Price Index (CPI) to normalize these values to a 2020 baseline. This adjustment ensured that older movies’ financial data, like those from the 1980s, held the same weight in our analysis as those from more recent years.

The process involved loading a dataset of CPI values, each associated with a particular year. We then calculated an “adjustment factor” for each year by dividing the CPI of the baseline year (2020) by the CPI of the movie’s release year. For example, if a movie was released in 1990, its budget and gross revenue values were multiplied by this adjustment factor, converting them to what they’d be in 2020 dollars. This adjustment allowed us to create new columns, budget_adjusted and gross_adjusted, replacing the original budget and revenue figures.

As a result, the dataset now contained inflation-adjusted budget and gross values, which offered a consistent metric for comparison across decades. This step was foundational, ensuring that any trends we identified weren’t skewed by the effects of inflation.

## Step 2: Feature Engineering – Creating Interaction Features
Once we had standardized financial data, we sought to improve the model’s predictive power by creating new “interaction features.” These features capture relationships between variables that might affect revenue, which single features alone cannot represent. Our goal was to identify patterns that were more complex than a straightforward correlation between budget and gross revenue.

For instance, we created an interaction feature combining the adjusted budget with runtime, labeled as budget_adjusted_times_runtime. This feature is designed to capture the impact of high-budget, long-duration movies, which may have a larger box office draw due to their scale and production quality. By combining budget and runtime, we create a variable that reflects the scale and potential impact of a movie as a large, lengthy production.

We also generated a feature combining adjusted budget with the release year (budget_adjusted_times_year). This interaction can highlight the effect of high-budget films over different periods, potentially capturing shifts in viewer demand or industry trends that make high-budget films more lucrative in certain years.

Another key feature was the ratio of adjusted budget to runtime, budget_adjusted_to_runtime_ratio, which reflects the budget efficiency per minute of the film. This feature helps analyze whether movies with higher production values per minute tend to perform better.

The thought process here was to look beyond simple relationships and instead capture richer, combined metrics that may better correlate with revenue. By creating these interaction features, we aimed to increase our model’s ability to detect patterns and predict outcomes based on a more nuanced view of each movie’s characteristics.

## Step 3: Aggregated Historical Features
Next, we introduced aggregation features to encapsulate the historical impact of key contributors like writers, directors, and stars. These aggregations are mean revenue calculations, representing the average gross revenue associated with each writer, director, or star based on their past projects. For example, writer_mean_gross provides the mean revenue of all movies written by a particular writer, while director_mean_gross does the same for directors.

The reasoning behind this approach is that individuals with a track record of success are likely to continue driving revenue. For instance, a writer with multiple high-grossing films would have a high writer_mean_gross, indicating that their involvement might be a strong indicator of a movie’s financial success. By creating aggregated features, we effectively “summarize” each contributor’s historical performance, allowing our model to make predictions based on established reputations rather than isolated metrics.

In terms of execution, we used the grouped historical gross revenue values associated with each contributor to generate these averages. These new columns added valuable insights to our dataset, reflecting the accumulated success of each contributor and capturing their potential influence on revenue.

## Step 4: Correlation Analysis and Feature Selection
After creating interaction and aggregation features, we performed a comprehensive correlation analysis to identify the most predictive features. Using Pearson correlation, which measures the strength of linear relationships, we evaluated each feature’s correlation with the target variable gross_adjusted. By observing these correlations, we could filter out weaker predictors and focus on variables that held significant predictive power.

The analysis revealed the following key findings:

budget_adjusted_times_runtime: This feature had a strong correlation of 0.69 with gross_adjusted, suggesting that high-budget, longer films tend to achieve greater box office success. This metric was essential, capturing the impact of large-scale, high-production movies.

budget_adjusted_times_year: Showing a correlation of 0.68, this feature indicated that certain high-budget movies were particularly successful based on their release period, reflecting industry trends and audience expectations over time.

budget_adjusted_to_year_ratio: With a correlation of 0.68, this feature highlighted that the relative size of a movie’s budget compared to the typical budgets of the era was an influential factor, underscoring how larger budgets in certain years could drive revenue.

Among the aggregated features, writer_mean_gross and director_mean_gross stood out with correlations of 0.78 and 0.72, respectively. These high values reaffirmed that the historical performance of a movie’s writer and director are crucial indicators of a movie’s financial prospects. Writers and directors with consistent box-office success are likely to bring strong audience appeal, translating into higher revenues.

After this analysis, we prioritized features with correlations above 0.65, such as the interaction and aggregation metrics mentioned above, while de-prioritizing features with weaker correlations, such as runtime or release month interactions, which showed low to negligible relationships with revenue. This prioritization process ensured our model remained focused on the most predictive, meaningful indicators.

## Step 5: Visualization and Data Insights
To complement our correlation findings, we created visualizations that allowed us to examine the data patterns more closely. Histograms were used to display the frequency distributions of key variables like adjusted budget and runtime, while scatter plots showed the relationships between gross_adjusted and influential predictors, such as budget_adjusted_times_runtime. These visualizations helped us confirm the correlations and observe trends within the data.

For instance, scatter plots of budget_adjusted_times_runtime vs. gross_adjusted revealed a clear trend: as the interaction value increased, so did revenue, underscoring the importance of this feature. Heatmaps were used to consolidate and display correlation values for both numerical and categorical variables, visually highlighting the strength of key relationships. The visualizations provided additional evidence supporting our feature selection choices, reinforcing that the prioritized features had strong, observable connections with revenue.
![image](https://github.com/user-attachments/assets/fa17090b-c6ee-4864-87c1-be15fb8b32c6)
![image](https://github.com/user-attachments/assets/9c10fa8a-62ab-42f1-b29e-99e82c9f0d32)
![image](https://github.com/user-attachments/assets/6b5d8145-8f08-4eff-872c-9a4948e20a4f)

Final Dataset and Results
After these stages, we finalized a dataset that included the most predictive pre-release features, standardized financial data, and relevant interaction and aggregation features. This dataset reflected a comprehensive approach to feature engineering and selection, grounded in the analysis of complex relationships between variables.

The resulting model-ready dataset focused exclusively on realistic, pre-release indicators, offering a foundation for accurate, actionable revenue predictions. By carefully designing each feature and thoroughly analyzing its relevance, we ensured that our model could forecast revenue based on solid, data-driven insights without relying on post-release metrics, which could introduce bias or data leakage.
![image](https://github.com/user-attachments/assets/e3b1d982-026b-4b91-90fc-c79088b67caa)
![image](https://github.com/user-attachments/assets/ca7ac55c-0ecc-44b4-90b0-11b94574047c)
![image](https://github.com/user-attachments/assets/c761ce68-c0d9-452b-aa2a-c1783758401d)

## Step 6: Baseline Linear Regression Model
To establish an understanding of movie revenue prediction, we implemented linear regression as our baseline modeling approach. We chose to do linear regression because it provides a straightforward, interpretable framework for assessing the relationships between features and revenue. This baseline model allowed us to evaluate the predictive power of our engineered features before considering more complex methods.

Our primary goal was to see if a simple, linear model could capture the main drivers of revenue identified in our feature engineering and selection process. Features like writer_mean_gross and director_mean_gross showed high correlations with revenue, and we hypothesized that these variables, combined with our interaction terms, could yield strong predictive power even in a linear model.


Feature Selection and Preparation: Based on our previous correlation analysis, we focused on high-correlation features, including interaction features such as budget_adjusted_times_runtime and historical performance metrics like writer_mean_gross. These features were prioritized for their strong linear relationships with gross_adjusted, our target variable.

Training the Model: With the selected features, we trained a linear regression model using gross_adjusted as the target. Linear regression allowed us to assess how well these variables could predict revenue without introducing additional complexity. This step served as a benchmark for understanding how much variance in revenue could be explained by the engineered features alone.


R² Score: The model achieved an R² score of 0.73, indicating that it explains 73% of the variance in movie revenue. This score suggests that our selected features capture a good portion of revenue patterns, validating the importance of variables like writer_mean_gross and budget_adjusted_times_runtime.

Mean Squared Error (MSE): The MSE was high, but this is expected due to the large scale of revenue values (often in millions or billions). This high MSE should be interpreted in context: it reflects the magnitude of the target variable rather than suggesting poor model accuracy. In this case, R² is a more meaningful indicator of the model’s effectiveness.

The linear regression model provided a useful baseline, confirming that our selected features have strong predictive relationships with revenue. However, the limitations of a linear model became evident, as some patterns in the data may be better captured by non-linear methods. 

## Step 7: Decision Tree Model
To advance from a simple baseline and better understand the feature dynamics affecting movie revenue, we implemented a Decision Tree model. We chose this method for its interpretability and ability to capture non-linear relationships within the data. The decision tree's structure of if-then rules allowed us to observe the primary factors driving revenue and how features interact in hierarchical ways.

Feature Selection and Preparation: Based on the results of our feature engineering, we utilized features with high correlations to revenue, such as budget_adjusted, runtime, release_month, and mean-encoded variables like genre_mean_gross. This feature selection aimed to identify key splits in the data where the variance in movie revenue is maximally reduced, a principle inherent to decision tree algorithms.

Training the Model: We trained the Decision Tree model using the gross_adjusted as our target variable. The tree structure provided clear insights into which features were most influential in predicting revenue. While the model captured important patterns, it was prone to overfitting due to the nature of decision trees, which can memorize the training data when left unchecked.

Model Evaluation: 
R² Score: The Decision Tree model achieved an R² score of 0.67, explaining 67% of the variance in movie revenue. While this demonstrates a moderate level of predictive power, it highlights the limitations of a single tree in handling the complexity of the data.
RMSE: The Root Mean Squared Error was 122,363,329.35, reflecting a substantial average prediction error. This high RMSE is expected, given the wide range of movie revenue figures and the decision tree’s tendency to create rigid splits that may not generalize well.

Insights and Limitations: The Decision Tree model validated the importance of budget and other high-correlation features in predicting revenue. However, the model struggled with generalization and was sensitive to variations in the data, indicating the need for more robust ensemble methods to reduce overfitting and improve accuracy.

## Step 8: Random Forest Model
Building on the insights from the Decision Tree model, we implemented a Random Forest model to enhance prediction accuracy and mitigate overfitting. We selected Random Forest because it combines multiple decision trees to create an ensemble that is more robust and capable of capturing complex patterns in the data.

Feature Selection and Preparation: The features selected for the Random Forest model were the same high-impact features used previously, such as budget_adjusted, runtime, release_month, and mean-encoded categorical variables. By using multiple trees and averaging their predictions, the Random Forest model capitalized on the strengths of each individual tree while reducing the variance associated with overfitting.

Training the Model: We trained the Random Forest model with 100 decision trees and incorporated bootstrapping to ensure diverse and stable predictions. The ensemble approach allowed the model to learn a broader range of patterns, improving generalization and prediction accuracy. This method also helped capture interactions between features that a single decision tree would miss.

Model Evaluation: 
R² Score: The Random Forest model achieved an impressive R² score of 0.82, explaining 82% of the variance in movie revenue. This score represents a significant improvement over the Decision Tree model, indicating that the ensemble approach effectively captures more complex patterns in the data.
RMSE: The RMSE was 89,611,379.28, significantly lower than that of the Decision Tree model. Although the error remains high due to the scale of movie revenue, the reduction in RMSE demonstrates that Random Forest offers more accurate and reliable predictions.

Outcome: The Random Forest model successfully addressed the overfitting issues of the Decision Tree, providing a more accurate and generalizable prediction framework. The ensemble’s ability to capture non-linear interactions and reduce variance made it a strong contender for movie revenue prediction. However, there is still room for improvement, especially in handling outliers and extreme revenue values, which could benefit from additional feature engineering or advanced methods like boosting.

# Proposal:
--------------
## Description:
This project will develop a predictive model that can forecast the box office success of upcoming movies based on various factors (cast, director, budget, genre, release date, etc.). The model will provide insights into the key elements that contribute to a movie’s box office performance. The data will be collected from **The Movie Database (TMDb) API**.

## Clear Goal:
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
