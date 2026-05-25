# Blood Pressure Abnormality Prediction Using Machine Learning

# Project Overview

This project is a Machine Learning-based healthcare prediction system developed to predict whether a patient has abnormal blood pressure using health and lifestyle-related data.
The system analyzes patient information such as:
Age
BMI
Stress Level
Alcohol Consumption
Pregnancy
Genetic Factors
and predicts blood pressure abnormality using the Random Forest Classifier algorithm.

# Objectives

Predict abnormal blood pressure using machine learning.
Perform healthcare data analysis and preprocessing.
Handle missing values and duplicate records.
Visualize important health factors affecting blood pressure.
Build an accurate and efficient prediction model.

# Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook

# Machine Learning Algorithm

Random Forest Classifier
Used for classification and prediction of blood pressure abnormality.

#  Project Workflow

1. Import Libraries
Imported Python libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
2. Load Dataset
Loaded the healthcare CSV dataset into Python using Pandas.
3. Data Cleaning and Preprocessing
Cleaned the dataset and prepared it for machine learning.
4. Handle Missing Values
Filled missing data using suitable methods like median and zero filling.
5. Remove Duplicates
Removed duplicate records to improve data quality.
6. Exploratory Data Analysis (EDA)
Analyzed data patterns and relationships using graphs and charts.
7. Handle Class Imbalance using Oversampling
Balanced the dataset by increasing minority class samples.
8. Train-Test Split
Split the dataset into training and testing data.
9. Train Random Forest Model
Trained the Random Forest Classifier for prediction.
10. Evaluate Model Performance
Checked model accuracy using confusion matrix and classification metrics.
11. Save Model using Pickle
Saved the trained machine learning model as a .pkl file for future use.


# Visualizations
#  Visualization 1 – Box Plot

Age vs Blood Pressure Abnormality
This box plot compares age distribution with blood pressure abnormality.
Observation:
Patients with abnormal blood pressure generally belong to higher age groups.
Median age is higher for abnormal cases.
Conclusion:
Age is an important risk factor for blood pressure abnormalities.


#  Visualization 2 – Count Plot

Gender vs Blood Pressure Abnormality
This graph shows male and female patient counts.
Observation:
Male patients had slightly more abnormal blood pressure cases.
Gender influences blood pressure patterns.
Conclusion:
Gender can be considered an important feature for prediction.

# Visualization 3 – Stress Level Analysis

This visualization shows stress level impact on blood pressure.
Observation:
Higher stress levels show more abnormal blood pressure cases.
Conclusion:
Stress is strongly associated with hypertension risk.

# Visualization 4 – Correlation Heatmap

Now let us discuss the heatmap.
A heatmap shows relationships between features.
Observation:
Age,
BMI,
and Stress Level
have strong positive correlation with blood pressure abnormality.
This means these features influence prediction significantly.
Heatmap helps us identify:
strong features,
weak features,
and feature relationships.


# Conclusion

This project demonstrates how machine learning can support healthcare systems through early prediction of blood pressure abnormalities. Early detection helps reduce health risks and improves medical decision-making.
