# CS506 Final Project: Box Office Revenue Prediction
**Final Project Report: Movie Revenue Prediction Analysis**
---

### Youtube Link: 
https://youtu.be/IGwYjbQEvsE

### Team:
- Sofia Boada: sofiabl@bu.edu
- Ana Julia Bortolossi: anajbdbl@bu.edu
- Suraj Godithi: surajg@bu.edu
- Francisco Moguel: fjmoguel@bu.edu 

**Data Sources:**
- **Kaggle Datasets**: [Movies Dataset](https://www.kaggle.com/datasets/danielgrijalvas/movies)

### Introduction

The objective of this project was to build a machine learning model capable of predicting movie gross revenues using a range of features. These included budget, runtime, release month, and interaction variables. The results of the analysis provided insights into how key factors contribute to movie revenue, supported by a series of visualizations and statistical evaluations. Once the model was fit into the data, we generated an interactive webpage that is capable of forecasting the gross revenus the given movie (with the input parameters).

---

### Code and Reproducibility

#### **Makefile Overview**

The provided `Makefile` serves as an essential component to ensure reproducibility, ease of use, and clarity in executing the project. Below are the key functionalities it provides:

1. **`install`**: Creates a virtual environment using `venv`, upgrades `pip`, and installs the required dependencies listed in `requirements.txt`. This ensures the environment is consistent across different systems.
   ```bash
   make install
   ```

2. **`run`**: Activates the virtual environment and launches the Flask web application for the movie revenue predictor. The app runs on `http://0.0.0.0:5001`.
   ```bash
   make run
   ```

3. **`clean`**: Removes the virtual environment entirely, ensuring a fresh start for new installations.
   ```bash
   make clean
   ```

4. **`freeze`**: Captures the exact dependencies and their versions into a `requirements.txt` file, aiding in reproducibility.
   ```bash
   make freeze
   ```

#### **Web Application Code Structure**

The web application is built using Flask, with a primary entry point defined in `app.py`. Key components are outlined below:

1. **Model Integration**:
   - The pre-trained XGBoost model is loaded using `xgb.XGBRegressor`.
   - Predictions are made by processing user inputs (e.g., budget, runtime, release month) and applying transformations to generate interaction variables such as `budget_adjusted_times_runtime`. We tried ensamble and linear midels beforehand, however, boosting provided the best tradeoff in accuracy and runtime to build our app upon.

2. **Endpoints**:
   - **`GET`**: Renders the main HTML page (`index.html`) where users can input movie details. It also provides an explanation of why the result happened and why. It also displays a variable importance graph. 
   - **`POST`**: Handles form submissions, calculates predictions, confidence intervals, and generates explanations based on user inputs.

3. **Supporting Functions**:
   - **`get_feature_importance`**: Generates a bar chart of feature importance from the XGBoost model.
   - **`calculate_confidence_interval`**: Computes simplistic confidence intervals (+/-10% margin) around the predicted revenue.
   - **`generate_explanation`**: Constructs natural-language explanations based on the user's input (e.g., highlighting the impact of budget or runtime).

#### **Frontend Implementation**

The frontend is a responsive web page built with **TailwindCSS** and custom JavaScript for user interactivity. Key sections include:

1. **Input Form**:
   - Users can enter details such as the movie's year of release, budget, runtime, and release month.
   - Dropdowns and placeholders guide users on expected inputs.

2. **Prediction Results**:
   - Predicted revenue in both release-year and 2020-adjusted dollars is displayed.
   - Confidence intervals are prominently shown to provide uncertainty estimates.

3. **Feature Importance Visualization**:
   - A dynamically generated feature importance graph (base64-encoded PNG) is embedded directly into the webpage.

#### **Tests and Validation**

Unit tests are provided in `tests/test.py` to validate the following:

- Correct loading and initialization of the XGBoost model.
- Proper calculation of inflation adjustment factors using CPI data.
- Accurate prediction generation and comparison with known test cases.
- Reproducibility of feature importance scores across runs.

#### **Dependencies**

The project relies on the following core dependencies, detailed in `requirements.txt`:

- **Flask**: For web application development.
- **pandas**: For data manipulation and preprocessing.
- **xgboost**: For model training and prediction.
- **numpy**: For numerical operations.
- **matplotlib**: For visualizing feature importances.
- **scikit-learn**: For evaluation metrics and additional model utilities.

##### Midterm Report Youtube Link:
https://youtu.be/Uk0iqkvShx0

### Initial Steps before final application

##### Step 1: Data Preparation and Inflation Adjustment
To create a reliable model, we began by standardizing financial data over time, as movies in the dataset span multiple decades. Since inflation affects both the meaning and value of financial figures like budgets and revenue, we couldn’t directly compare these amounts across different years without adjustment. We used the Consumer Price Index (CPI) to normalize these values to a 2020 baseline. This adjustment ensured that older movies’ financial data, like those from the 1980s, held the same weight in our analysis as those from more recent years.

The process involved loading a dataset of CPI values, each associated with a particular year. We then calculated an “adjustment factor” for each year by dividing the CPI of the baseline year (2020) by the CPI of the movie’s release year. For example, if a movie was released in 1990, its budget and gross revenue values were multiplied by this adjustment factor, converting them to what they’d be in 2020 dollars. This adjustment allowed us to create new columns, budget_adjusted and gross_adjusted, replacing the original budget and revenue figures.

As a result, the dataset now contained inflation-adjusted budget and gross values, which offered a consistent metric for comparison across decades. This step was foundational, ensuring that any trends we identified weren’t skewed by the effects of inflation.

##### Step 2: Feature Engineering – Creating Interaction Features
Once we had standardized financial data, we sought to improve the model’s predictive power by creating new “interaction features.” These features capture relationships between variables that might affect revenue, which single features alone cannot represent. Our goal was to identify patterns that were more complex than a straightforward correlation between budget and gross revenue.

For instance, we created an interaction feature combining the adjusted budget with runtime, labeled as budget_adjusted_times_runtime. This feature is designed to capture the impact of high-budget, long-duration movies, which may have a larger box office draw due to their scale and production quality. By combining budget and runtime, we create a variable that reflects the scale and potential impact of a movie as a large, lengthy production.

We also generated a feature combining adjusted budget with the release year (budget_adjusted_times_year). This interaction can highlight the effect of high-budget films over different periods, potentially capturing shifts in viewer demand or industry trends that make high-budget films more lucrative in certain years.

Another key feature was the ratio of adjusted budget to runtime, budget_adjusted_to_runtime_ratio, which reflects the budget efficiency per minute of the film. This feature helps analyze whether movies with higher production values per minute tend to perform better.

The thought process here was to look beyond simple relationships and instead capture richer, combined metrics that may better correlate with revenue. By creating these interaction features, we aimed to increase our model’s ability to detect patterns and predict outcomes based on a more nuanced view of each movie’s characteristics.

##### Step 3: Aggregated Historical Features
Next, we introduced aggregation features to encapsulate the historical impact of key contributors like writers, directors, and stars. These aggregations are mean revenue calculations, representing the average gross revenue associated with each writer, director, or star based on their past projects. For example, writer_mean_gross provides the mean revenue of all movies written by a particular writer, while director_mean_gross does the same for directors.

The reasoning behind this approach is that individuals with a track record of success are likely to continue driving revenue. For instance, a writer with multiple high-grossing films would have a high writer_mean_gross, indicating that their involvement might be a strong indicator of a movie’s financial success. By creating aggregated features, we effectively “summarize” each contributor’s historical performance, allowing our model to make predictions based on established reputations rather than isolated metrics.

In terms of execution, we used the grouped historical gross revenue values associated with each contributor to generate these averages. These new columns added valuable insights to our dataset, reflecting the accumulated success of each contributor and capturing their potential influence on revenue.

##### Step 4: Correlation Analysis and Feature Selection
After creating interaction and aggregation features, we performed a comprehensive correlation analysis to identify the most predictive features. Using Pearson correlation, which measures the strength of linear relationships, we evaluated each feature’s correlation with the target variable gross_adjusted. By observing these correlations, we could filter out weaker predictors and focus on variables that held significant predictive power.

The analysis revealed the following key findings:

budget_adjusted_times_runtime: This feature had a strong correlation of 0.69 with gross_adjusted, suggesting that high-budget, longer films tend to achieve greater box office success. This metric was essential, capturing the impact of large-scale, high-production movies.

budget_adjusted_times_year: Showing a correlation of 0.68, this feature indicated that certain high-budget movies were particularly successful based on their release period, reflecting industry trends and audience expectations over time.

budget_adjusted_to_year_ratio: With a correlation of 0.68, this feature highlighted that the relative size of a movie’s budget compared to the typical budgets of the era was an influential factor, underscoring how larger budgets in certain years could drive revenue.

Among the aggregated features, writer_mean_gross and director_mean_gross stood out with correlations of 0.78 and 0.72, respectively. These high values reaffirmed that the historical performance of a movie’s writer and director are crucial indicators of a movie’s financial prospects. Writers and directors with consistent box-office success are likely to bring strong audience appeal, translating into higher revenues.

After this analysis, we prioritized features with correlations above 0.65, such as the interaction and aggregation metrics mentioned above, while de-prioritizing features with weaker correlations, such as runtime or release month interactions, which showed low to negligible relationships with revenue. This prioritization process ensured our model remained focused on the most predictive, meaningful indicators.
![image](https://github.com/user-attachments/assets/bb379551-2963-426e-8802-738665bae2c3)
![image](https://github.com/user-attachments/assets/6f5052c5-bc8e-4834-bb98-93a753ba55bc)
![image](https://github.com/user-attachments/assets/c711a44d-5a05-4f08-9d5c-d4c74d52a523)
![image](https://github.com/user-attachments/assets/b34fbd6a-5c0e-4a2f-a651-1a0453b4260e)

##### Step 5: Visualization and Data Insights
To complement our correlation findings, we created visualizations that allowed us to examine the data patterns more closely. Histograms were used to display the frequency distributions of key variables like adjusted budget and runtime, while scatter plots showed the relationships between gross_adjusted and influential predictors, such as budget_adjusted_times_runtime. These visualizations helped us confirm the correlations and observe trends within the data.

For instance, scatter plots of budget_adjusted_times_runtime vs. gross_adjusted revealed a clear trend: as the interaction value increased, so did revenue, underscoring the importance of this feature. Heatmaps were used to consolidate and display correlation values for both numerical and categorical variables, visually highlighting the strength of key relationships. The visualizations provided additional evidence supporting our feature selection choices, reinforcing that the prioritized features had strong, observable connections with revenue.


Final Dataset and Results
After these stages, we finalized a dataset that included the most predictive pre-release features, standardized financial data, and relevant interaction and aggregation features. This dataset reflected a comprehensive approach to feature engineering and selection, grounded in the analysis of complex relationships between variables.

The resulting model-ready dataset focused exclusively on realistic, pre-release indicators, offering a foundation for accurate, actionable revenue predictions. By carefully designing each feature and thoroughly analyzing its relevance, we ensured that our model could forecast revenue based on solid, data-driven insights without relying on post-release metrics, which could introduce bias or data leakage.
![image](https://github.com/user-attachments/assets/e3b1d982-026b-4b91-90fc-c79088b67caa)
![image](https://github.com/user-attachments/assets/ca7ac55c-0ecc-44b4-90b0-11b94574047c)
![image](https://github.com/user-attachments/assets/c761ce68-c0d9-452b-aa2a-c1783758401d)

##### Step 6: Baseline Linear Regression Model
To establish an understanding of movie revenue prediction, we implemented linear regression as our baseline modeling approach. We chose to do linear regression because it provides a straightforward, interpretable framework for assessing the relationships between features and revenue. This baseline model allowed us to evaluate the predictive power of our engineered features before considering more complex methods.

Our primary goal was to see if a simple, linear model could capture the main drivers of revenue identified in our feature engineering and selection process. Features like writer_mean_gross and director_mean_gross showed high correlations with revenue, and we hypothesized that these variables, combined with our interaction terms, could yield strong predictive power even in a linear model.


Feature Selection and Preparation: Based on our previous correlation analysis, we focused on high-correlation features, including interaction features such as budget_adjusted_times_runtime and historical performance metrics like writer_mean_gross. These features were prioritized for their strong linear relationships with gross_adjusted, our target variable.

Training the Model: With the selected features, we trained a linear regression model using gross_adjusted as the target. Linear regression allowed us to assess how well these variables could predict revenue without introducing additional complexity. This step served as a benchmark for understanding how much variance in revenue could be explained by the engineered features alone.


R² Score: The model achieved an R² score of 0.73, indicating that it explains 73% of the variance in movie revenue. This score suggests that our selected features capture a good portion of revenue patterns, validating the importance of variables like writer_mean_gross and budget_adjusted_times_runtime.

Mean Squared Error (MSE): The MSE was high, but this is expected due to the large scale of revenue values (often in millions or billions). This high MSE should be interpreted in context: it reflects the magnitude of the target variable rather than suggesting poor model accuracy. In this case, R² is a more meaningful indicator of the model’s effectiveness.

The linear regression model provided a useful baseline, confirming that our selected features have strong predictive relationships with revenue. However, the limitations of a linear model became evident, as some patterns in the data may be better captured by non-linear methods. 

##### Step 7: Decision Tree Model
To advance from a simple baseline and better understand the feature dynamics affecting movie revenue, we implemented a Decision Tree model. We chose this method for its interpretability and ability to capture non-linear relationships within the data. The decision tree's structure of if-then rules allowed us to observe the primary factors driving revenue and how features interact in hierarchical ways.

Feature Selection and Preparation: Based on the results of our feature engineering, we utilized features with high correlations to revenue, such as budget_adjusted, runtime, release_month, and mean-encoded variables like genre_mean_gross. This feature selection aimed to identify key splits in the data where the variance in movie revenue is maximally reduced, a principle inherent to decision tree algorithms.

Training the Model: We trained the Decision Tree model using the gross_adjusted as our target variable. The tree structure provided clear insights into which features were most influential in predicting revenue. While the model captured important patterns, it was prone to overfitting due to the nature of decision trees, which can memorize the training data when left unchecked.

Model Evaluation: 
R² Score: The Decision Tree model achieved an R² score of 0.67, explaining 67% of the variance in movie revenue. While this demonstrates a moderate level of predictive power, it highlights the limitations of a single tree in handling the complexity of the data.
RMSE: The Root Mean Squared Error was 122,363,329.35, reflecting a substantial average prediction error. This high RMSE is expected, given the wide range of movie revenue figures and the decision tree’s tendency to create rigid splits that may not generalize well.

Insights and Limitations: The Decision Tree model validated the importance of budget and other high-correlation features in predicting revenue. However, the model struggled with generalization and was sensitive to variations in the data, indicating the need for more robust ensemble methods to reduce overfitting and improve accuracy.

##### Step 8: Random Forest Model
Building on the insights from the Decision Tree model, we implemented a Random Forest model to enhance prediction accuracy and mitigate overfitting. We selected Random Forest because it combines multiple decision trees to create an ensemble that is more robust and capable of capturing complex patterns in the data.

Feature Selection and Preparation: The features selected for the Random Forest model were the same high-impact features used previously, such as budget_adjusted, runtime, release_month, and mean-encoded categorical variables. By using multiple trees and averaging their predictions, the Random Forest model capitalized on the strengths of each individual tree while reducing the variance associated with overfitting.

Training the Model: We trained the Random Forest model with 100 decision trees and incorporated bootstrapping to ensure diverse and stable predictions. The ensemble approach allowed the model to learn a broader range of patterns, improving generalization and prediction accuracy. This method also helped capture interactions between features that a single decision tree would miss.

Model Evaluation: 
R² Score: The Random Forest model achieved an impressive R² score of 0.82, explaining 82% of the variance in movie revenue. This score represents a significant improvement over the Decision Tree model, indicating that the ensemble approach effectively captures more complex patterns in the data.
RMSE: The RMSE was 89,611,379.28, significantly lower than that of the Decision Tree model. Although the error remains high due to the scale of movie revenue, the reduction in RMSE demonstrates that Random Forest offers more accurate and reliable predictions.

Outcome: The Random Forest model successfully addressed the overfitting issues of the Decision Tree, providing a more accurate and generalizable prediction framework. The ensemble’s ability to capture non-linear interactions and reduce variance made it a strong contender for movie revenue prediction. However, there is still room for improvement, especially in handling outliers and extreme revenue values, which could benefit from additional feature engineering or advanced methods like boosting.

-----------------------------------------------------

### Data Overview after early steps

The dataset contained historical data on movies, including their adjusted budgets, runtimes, gross adjusted revenues, release months, and interaction variables derived from these base features. The key features were:

1. **Budget Adjusted**: The inflation-adjusted production budget of a movie.
2. **Runtime**: The duration of the movie in minutes.
3. **Release Month**: The month the movie was released, represented numerically.
4. **Interaction Variables**: Multiplicative and ratio-based features such as:
   - `budget_adjusted_times_runtime`
   - `budget_adjusted_times_year`
   - `budget_adjusted_to_year_ratio`
   - `budget_adjusted_to_runtime_ratio`

To enhance interpretability, we focused on features with correlations above 0.65 with the target variable (`gross_adjusted`). These included `budget_adjusted_times_runtime`, `budget_adjusted_times_year`, and `budget_adjusted_to_year_ratio`.

---

### Model Overview

We used an XGBoost Regressor trained on key features to predict `gross_adjusted`. The model incorporated:

- Feature engineering with interaction variables.
- Data cleaning to ensure no missing values in critical columns.
- Hyperparameter tuning to optimize performance.

The model achieved:

- **Root Mean Squared Error (RMSE)**: 101,991,097.67
- **R² Score**: 0.78

---

### Key Visualizations and Insights

#### 1. **3D Scatter Plot**

![image](https://github.com/user-attachments/assets/e2988a3e-ad15-400c-9c16-745894e41252)

The 3D scatter plot of runtime, predicted gross adjusted, and actual gross adjusted highlighted the clustering of movies with longer runtimes and higher predicted revenues. This visualization demonstrated the model's performance across different revenue ranges.

#### 2. **Average Runtime and Budget by Year**

![image](https://github.com/user-attachments/assets/8d1bbd9c-c746-4394-9124-a09578c528c5)

This graph illustrated an upward trend in both average runtime and budget over the years, with notable spikes around major blockbuster periods. The increasing production budgets aligned with trends in rising movie revenue potential.

#### 3. **Yearly Trend of Gross Adjusted**

![image](https://github.com/user-attachments/assets/bcdffdd4-bf20-480f-999d-01a0951d56c1)

This visualization showed a steady increase in average gross adjusted revenues over the decades. Peaks in recent years reflected the dominance of high-budget productions. It also displays how movies have higher investment in much more marketing than in the past. Finally, after investigation, it also helps on how movies have higher revenues in the internatational box office than in past decades. 

#### 4. **Prediction Variance by Key Features**

![image](https://github.com/user-attachments/assets/74f83a6b-22ee-4e69-9d40-56363e343be2)

A plot of variance in predictions against feature values revealed significant variability in movies with extreme budgets or runtimes. This suggested the potential for outlier-driven effects in high-budget productions.

#### 5. **Permutation Importance Heatmap**

![image](https://github.com/user-attachments/assets/31f72058-2e0a-439e-b031-ab8cc6861337)

The heatmap revealed that `budget_adjusted` and `budget_adjusted_times_runtime` were the most influential features in predicting movie revenues, reinforcing the importance of financial investment in movie success.

#### 6. **Key Feature Pairplot**

![image](https://github.com/user-attachments/assets/52eab49b-3a26-4062-9f1f-83e0c65ee722)

Pairplot visualizations highlighted strong positive relationships between `budget_adjusted` and `gross_adjusted`. A slight non-linear relationship with runtime was also evident, where optimal runtimes contributed positively to revenues.

#### 7. **Learning Curve**

![image](https://github.com/user-attachments/assets/d427bc80-f5b8-4897-95aa-5da482124ccb)

The learning curve revealed that the model generalized well with increasing training data. While the train score was consistently high, test scores improved with more data, reflecting effective feature utilization.

#### 8. **Hyperparameter Tuning Scores**

![image](https://github.com/user-attachments/assets/100856c3-4da9-49b7-973c-5f27221587da)

The hyperparameter tuning graph demonstrated the importance of balancing training and testing scores. Increasing parameter values improved model performance up to a threshold, after which overfitting became evident.

#### 9. **Confidence Intervals by Release Month**

![image](https://github.com/user-attachments/assets/d0e62a0a-e70e-4a74-adf6-6fb7d3db532d)

Confidence intervals for predictions across release months showed higher revenues in June, July, and December, aligning with industry trends for summer blockbusters and holiday releases.

#### 10. **Distribution of Prediction Errors**

![image](https://github.com/user-attachments/assets/d63a7b3b-edbd-4bd5-9501-3b407cfa59be)

The error distribution, centered around zero with minimal skew, confirmed the model’s predictive accuracy and robustness. It shows slight underprediction across the model.

---

### Statistical Evaluation

Correlation analyses confirmed the importance of interaction variables. For instance:

- `budget_adjusted_times_runtime`: Correlation with `gross_adjusted` = 0.69
- `budget_adjusted_times_year`: Correlation with `gross_adjusted` = 0.68
- `writer_mean_gross`: Correlation with `gross_adjusted` = 0.78

These features were instrumental in driving model performance.

The correlation analysis highlights the significant impact of interaction variables in predicting movie revenues. The strong correlation between budget_adjusted_times_runtime (0.69) and gross_adjusted suggests that higher budgets combined with optimal runtimes are a critical determinant of box office success. This could imply that well-funded movies with carefully balanced runtimes resonate better with audiences, achieving higher revenues. Similarly, the correlation of budget_adjusted_times_year (0.68) with gross_adjusted indicates that the interplay between production budgets and the release year contributes to revenue trends, possibly reflecting inflation adjustments and evolving audience preferences. Furthermore, the writer_mean_gross correlation (0.78) underscores the importance of creative talent in driving revenue, as successful writers with a track record of high-grossing films likely bring credibility and audience anticipation to new projects. These insights affirm the role of strategic budgeting, runtime planning, and leveraging established creative talent in maximizing box office potential.

---

### Limitations and Future Work

1. **Data Coverage**: While the dataset was comprehensive, it lacked certain variables like international box office revenues or marketing spend, which could enhance predictions.
2. **Temporal Bias**: Older movies were underrepresented in terms of gross adjusted values due to incomplete data.
3. **Model Enhancements**: Future iterations could incorporate ensemble techniques or external economic indicators to improve accuracy.

---

### Final Remarks
This project has demonstrated the power of machine learning in understanding and predicting movie revenues, offering actionable insights for stakeholders in the film industry. By leveraging advanced techniques such as inflation adjustments, feature engineering, and ensemble modeling, we successfully created a robust tool capable of forecasting movie gross revenues with impressive accuracy. The integration of these techniques into an interactive web application has extended the utility of this project, making it a valuable resource for producers, analysts, and industry professionals.

One of the most significant findings was the role of interaction variables and aggregated historical features in predicting revenues. For instance, the interaction variable budget_adjusted_times_runtime revealed that high-budget movies with extended runtimes are strong drivers of box office success, reflecting industry trends favoring large-scale productions. Similarly, the feature budget_adjusted_times_year captured the interplay between budgets and release timing, highlighting the importance of strategic planning in aligning movie releases with favorable periods. The aggregated feature writer_mean_gross further emphasized the critical role of creative talent, with high-performing writers and directors significantly boosting revenue potential through audience anticipation and credibility.

The correlation analysis and feature importance rankings provided additional validation for these insights. Features like budget_adjusted, runtime, and their interactions consistently ranked as the most significant predictors of revenue, reinforcing the need for careful budgeting and runtime planning. Visualizations such as pairplots, heatmaps, and the learning curve added further depth to our understanding, illustrating clear trends and relationships in the data. For example, the learning curve demonstrated that the model generalized well with increasing training data, while the permutation importance heatmap highlighted the dominant role of financial investment in driving success.

Our iterative approach to model development, starting with linear regression and progressing through decision trees, random forests, and ultimately XGBoost, showcased the importance of selecting the right algorithm for the problem at hand. The XGBoost model stood out for its ability to balance accuracy and computational efficiency, achieving an R² score of 0.78 and a RMSE of 101,991,097.67. The use of hyperparameter tuning and SHAP-based explanations further enhanced the model's performance and interpretability, providing users with a deeper understanding of the factors influencing predictions.

The deployment of the model as a web application added practical value to the project. Built with Flask and TailwindCSS, the application features a user-friendly interface where users can input movie parameters and receive predictions along with confidence intervals and feature importance visualizations. This interactive tool not only provides predictions but also educates users on the underlying factors driving revenue, making it both practical and informative.

Despite its strengths, the project does have limitations. The dataset lacked variables such as international box office revenues and marketing expenditures, which could enhance predictions. Temporal biases were also evident, with older movies underrepresented in terms of adjusted gross values. Future iterations could address these gaps by incorporating external data sources, such as social media trends or global box office figures, and exploring advanced modeling techniques like deep learning or economic forecasting models.

Overall, this project highlights the critical role of data-driven decision-making in the film industry. From budget allocation to release timing and team selection, the insights generated by this model can guide producers in optimizing their strategies for maximum revenue potential. By combining advanced analytics, robust modeling techniques, and an intuitive web interface, this project not only delivers accurate predictions but also provides a framework for understanding the key drivers of box office success. It stands as a testament to the value of integrating machine learning into real-world applications, offering both predictive power and actionable insights for stakeholders.
