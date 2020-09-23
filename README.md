## Predicting Fuel Efficiency of Vehicles:
In this series, we'd be going from data collection to deploying the Machine Learning model:
1.	Data Collection - we are using the classic Auto MPG dataset from UCI ML Repository, I have already collected data and stored it in auto-mpg.data .
2.	Define Problem Statement - We'll frame the problem based on the dataset description and initial exploration.
3.	EDA - Carry our exploratory analysis to figure out the important features and creating new combination of features.
4.	Data Preparation - Using step 4, create a pipeline of tasks to transform the data to be loaded into our ML models.
5.	Selecting and Training ML models - Training a few models to evaluate their predictions using cross-validation.
6.	Hyperparameter Tuning - Fine tune the hyperparameters for the models that showed promising results.
7.	Saving Model: Using Pickle module we save our model.

Whole code is divided into two notebooks EDA.ipynb and model_prediction.ipynb. As, the name suggests , first notebook contains all the analysis of data required to make an idea and same is used in other notebook to build a model.

 
## Result:
We test our data on three models, Linear Regression, Decision Tree Regressor and Random Forest Regressor and latter was giving us the minimum error.
