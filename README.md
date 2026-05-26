# Machine-Learning-Deep-Learning-NLP-Applications
Titanic Passenger Survival Prediction Pipeline
This project builds a comprehensive machine learning and deep learning pipeline to predict passenger survival on the Titanic. It covers everything from initial data cleaning and handling class imbalance to training traditional ML classifiers and building an optimized Deep Neural Network (DNN) using TensorFlow/Keras.

 Project Overview
The goal of this project is to clean the classic Titanic dataset, engineer categorical features, scale the numerical vectors, and compare the classification performance of traditional models against a multi-layer Neural Network. Given the class imbalance present in the survival data, models are evaluated across multiple metrics, focusing heavily on the F1-Score.

 Tech Stack & Libraries
Data Manipulation: pandas, numpy

Data Visualization: matplotlib

Traditional Machine Learning: scikit-learn

Deep Learning Framework: tensorflow.keras

 Pipeline Steps
1. Data Cleaning & Preprocessing
Missing Values: * Age was filled using the median age to prevent outlier distortions.

Embarked was filled using the mode (most frequent gate).

Cabin missing elements were categorized under a new 'Unknown' flag.

Feature Dropping: Irrelevant columns such as PassengerId, Name, Ticket, and Cabin were omitted to ensure the model focuses purely on generalizable features.

Categorical Encoding: Sex and Embarked were converted to numerical representations using One-Hot Encoding (pd.get_dummies), utilizing drop_first=True to prevent multi-collinearity.

Feature Scaling: Input vectors were normalized using StandardScaler to ensure optimal performance for scale-sensitive algorithms like KNN, Logistic Regression, and Neural Networks.

2. Model Architecture
A. Traditional Machine Learning Models
Three distinct types of classic baseline classifiers were implemented:

Logistic Regression: Max iterations set to 1000 to guarantee gradient convergence.

Random Forest Classifier: An ensemble tree method left at defaults to capture non
