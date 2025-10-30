# Customer Churn Prediction using Machine Learning
This project predicts telecom customer churn by analyzing usage patterns and customer behavior. Utilizes data preprocessing, EDA, and machine learning models to identify high-risk customers and support proactive retention strategies.

# Project Overview
This repository contains a comprehensive data science project focused on developing an accurate and actionable machine learning model for predicting customer churn within a telecommunications business. Customer churn is a mission-critical metric, and this project's core value lies in its ability to proactively identify customers at high risk of attrition. By predicting churn, the company can deploy targeted, cost-effective retention campaigns (e.g., customized offers, improved support) before the customer decides to leave.
The project rigorously follows a complete data science lifecycle, beginning with raw data ingestion and concluding with a fully trained and evaluated classification model ready for deployment.

# 🎯 Goal
The primary objective is to engineer a highly reliable classification model that determines whether a given customer will churn (Yes) or not (No). This prediction is based on a holistic view of the customer, encompassing their demographic data, the specific services they utilize, and their transactional account history.

# 📈 Key Results
Metric	Value	Model Used	Justification

Overall Accuracy	**78.00%**	Logistic Regression (Baseline Classifier)	Good general prediction rate for both Churn and Non-Churn classes.

Precision (for Churn)	**~60%**		Measures the quality of positive predictions (minimizing False Positives).

Recall (for Churn)	**~70%**		Measures the model's ability to find all actual churners (minimizing False Negatives).

F1-Score	**~65%**		Harmonic mean of Precision and Recall, crucial for imbalanced classification problems.

The model achieves a robust **78%** overall accuracy on the held-out test set, demonstrating strong predictive capability for operational use.

# 📁 Dataset
The analysis uses a well-known, publicly available telecom dataset, providing rich feature vectors across various customer dimensions:

•	Demographics: Gender, Senior Citizen status, Partner presence, and Dependents status.

•	Services Subscribed: Features detailing usage of Phone Service, Multiple Lines, Internet Service (DSL, Fiber Optic), Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, and Streaming Movies.

•	Account Information: Key transactional features including customer Tenure (months of service), Contract type (Month-to-month, One year, Two year), Paperless Billing status, Payment Method, Monthly Charges, and Total Charges.

•	Target Variable: The binary outcome variable, Churn (Yes/No).

# 🛠️ Methodology and Key Steps
The project notebook (CustomerChurnPrediction.ipynb) documents a meticulous step-by-step process:

**1. Data Cleaning and Preprocessing**

•	Initial Inspection: Conducted initial checks on data types, unique values, and statistical summaries.

•	Missing Value Handling (TotalCharges): Identified that the TotalCharges column contained blank strings for new customers (tenure=0), leading to a data type mismatch. These 11 missing records were appropriately imputed with zero (as they represent no total charges yet) and the column was successfully converted from object to float type.

•	Target Encoding: The categorical target variable, Churn (Yes/No), was transformed into a binary numerical label (1 for Churn, 0 for No Churn) for model compatibility.

**2. Exploratory Data Analysis (EDA)**

•	Class Imbalance Check: Visualized the distribution of the target variable to confirm the imbalance (fewer churners than non-churners), guiding the choice of evaluation metrics (F1-Score and Recall over simple Accuracy).

•	Impact of Contract Type: Analyzed churn rate by Contract type, revealing that Month-to-month contract holders have a disproportionately high churn rate—a key insight for marketing intervention.

•	Correlation with Numerical Features: Used histograms and scatter plots to examine the relationship between Tenure, MonthlyCharges, and the likelihood of churning. Short-term customers and those with high monthly fees showed elevated churn risk.

**3. Feature Engineering and Scaling**

•	Categorical Feature Encoding: Applied One-Hot Encoding (via pd.get_dummies) to all nominal categorical features (e.g., Contract, PaymentMethod, InternetService). This prevents the model from assuming ordinal relationships between non-ordered categories.

•	Feature Scaling: Applied MinMaxScaler or StandardScaler to numerical features (Tenure, MonthlyCharges, TotalCharges). This normalization ensures that features with larger numerical ranges do not unduly influence the model's objective function, which is critical for optimization algorithms like those used in Logistic Regression.

**4. Model Training and Evaluation**

•	Data Split: The engineered dataset was partitioned into training and testing sets (70% Train, 30% Test) to ensure unbiased evaluation of the model's generalization ability.

•	Model Selection: Logistic Regression was chosen as the initial baseline model due to its interpretability, speed, and effectiveness in binary classification tasks.

•	Evaluation: The model's performance was thoroughly assessed using multiple metrics, moving beyond simple accuracy to focus on the business impact of misclassification.

# 📊 Key Output Graphs
The notebook generates several crucial visualizations to validate the data and the model:

**1. Churn Rate by Contract Type**

A powerful bar plot demonstrating the stark difference in churn rates among customers based on their contract commitment. This visualization is key to identifying the most vulnerable customer segments.

**2. Confusion Matrix**

A visual representation of the model's performance on the test data. It clearly displays:
•	True Positives (TP): Correctly predicted churners (Ideal for retention efforts).
•	True Negatives (TN): Correctly predicted non-churners.
•	False Positives (FP): Customers predicted to churn but who did not (Wasted marketing effort).
•	False Negatives (FN): Customers who churned but were predicted not to (Missed retention opportunity—the costliest error).
The matrix helps interpret the Recall (minimizing FNs) and Precision (minimizing FPs) metrics.



