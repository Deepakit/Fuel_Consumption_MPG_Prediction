{
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


 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Predicting Fuel Efficiency of Vehicles:\n",
    "In this series, we'd be going from data collection to deploying the Machine Learning model:\n",
    "1.\tData Collection - we are using the classic Auto MPG dataset from UCI ML Repository, I have already collected data and stored it in auto-mpg.data .\n",
    "2.\tDefine Problem Statement - We'll frame the problem based on the dataset description and initial exploration.\n",
    "3.\tEDA - Carry our exploratory analysis to figure out the important features and creating new combination of features.\n",
    "4.\tData Preparation - Using step 4, create a pipeline of tasks to transform the data to be loaded into our ML models.\n",
    "5.\tSelecting and Training ML models - Training a few models to evaluate their predictions using cross-validation.\n",
    "6.\tHyperparameter Tuning - Fine tune the hyperparameters for the models that showed promising results.\n",
    "7.\tSaving Model: Using Pickle module we save our model.\n",
    "\n",
    "Whole code is divided into two notebooks EDA.ipynb and model_prediction.ipynb. As, the name suggests , first notebook contains all the analysis of data required to make an idea and same is used in other notebook to build a model.\n",
    "\n",
    " \n",
    "## Result:\n",
    "We test our data on three models, Linear Regression, Decision Tree Regressor and Random Forest Regressor and latter was giving us the minimum error.\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
