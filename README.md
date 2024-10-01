# CS506-final-project
CS506 Final Project: Box Office Revenue Prediction

Description: This project will be a predictive model that can forecast the box office success of upcoming movies based on various factors (cast, director, budget, genre, release data, etc.) The model will provide insights into the important elements that contribute to a movie’s box office performance. The data will be collected from The Movie Database (TMDb) API.


Clear goal:

	- Predict the box office revenue for an upcoming movie before its release using historical data on similar movies
	- Analyze the importance of different features (ex. Cast, budget, genre, etc.) in determining box office success

 
Data Collection: 

	- Data Sources
		- The Movie Database (TMDb) API: Provides movie metadata such as budget, genre, cast, and director. 
		- Kaggle Datasets: https://www.kaggle.com/datasets/danielgrijalvas/movies
	- Features to Collect:
		- Cast
		- Director
		- Budget
		- Release Date
		- Genre

  
Data Modeling: How you plan on modeling the data 

	- For the box office revenue predictor, the goal is to predict a movie's box office revenue based on features such as cast, director, budget, release date, and genre. After defining the target 	variable, which is revenue, we will test several models starting with something like Linear Regression as a baseline and progressing to more advanced models such as Decision Trees. We can 		evaluate the model's performance by using RMSE, or R^2 score.

 
Data Visualization: 

	- Model Visualiations
		- Predicted vs. Actual Revenue: Scatter plot to visualize the models accuracy
		- Feature Importance: Bar chart to show which factors most influence box office revenue
		- User Interface: allows for user to input details of a movie and predict its potential box office revenue
	- Data Analysis
		- Scatter plots comparing features like budget and revenue
		- Bar charts of feature correlations
		- Time-based visualizations showing revenue trends based on release date or season 

  
Test Plan:

	- Split the dataset into training and testing sets (80-20 or 70-30) to evaluate model performance on the unseen data
	- Model performance will be evaluated using metrics such as RMSE
	- The model will also be tested on movies that are released after training data ends to evaluate how well it generalizes to unseen data




​​









