# Churn prediction

## General info
The project concerns churn prediction in the bank customers. Based on data I have tried to predict whether the client is going to leave the bank or not by using information like credit score, tenure, salary, etc. Project includes data analysis, data preparation and created model by using different machine learning algorithms such Logistic Regression, Random Forest, KNN, Ada Boost and XGBoost.

### Dataset
The dataset contains the details of the customers in a bank company such as credit score, estimated salary, age, sex, etc. It comes from Kaggle and can be find 
[here](https://www.kaggle.com/shubh0799/churn-modelling).

## Motivation
The aim of the project were churn prediction in the bank customers. Churn is a term that means losing customers to the competition. A “Churned” customer is one who has cancelled their service and identification of such users beforehand can be invaluable from the company's point of view. It is very important because retain customers who want to leave us is in many cases much cheaper than acquiring new ones. In the analysis I have used different machine learning classifiers to predicted whether the client is going to leave the bank or not.

## Project contains:
- **Part 1: Exploratory Data Analysis** - Churn_EDA.ipynb
- **Part 2: Churn prediction in the bank customers** - Churn.ipynb
- Python scripts to train ML models - **churn_models.py, churn_best_model.py, helper_functions.py**
- models - models used in the project

## Summary
The goal of the project was churn prediction in the bank customers. I began with data analysis to better meet the data. Then I have cleaned it and prepared to the modelling. Firstly I have used five different classification models such as Logistic Regression, KNN, Random Forest, Ada Boost and XGBoost to achaived the best model. Following I have evaluated models with a few methods to check which model is the best. I have used a ROC AUC score, k-fold Cross Validation, ROC curve and confusion matrix. After checked all of this metrics the best classification algorithm that I got was Random Forest with ROC AUC score 85%. From the experiments one can see that It is a reasonably good model but it could be made a many improvement such as tuning the hyperparameter etc. to achaived a better results.

## Technologies
The project is created with:
- Python 3.6
- libraries: pandas, numpy, sklearn, seaborn, matplotlib, xgboost.

**Running the project:**

To run this project use Jupyter Notebook or Google Colab.

You can run the scripts in the terminal:

    churn_models.py
    churn_best_model.py
    helper_functions.py

