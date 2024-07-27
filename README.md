# Bank Loan Eligibility Classifier

This repository contains an analysis and classification model to predict loan approval status. Using historical loan application data, we have developed a machine learning model that can predict whether a loan will be approved or not. This project employs advanced machine learning techniques to ensure high predictive accuracy and reliability.

## Introduction and Objective
Predicting the approval of bank loans is a critical task for financial institutions as it directly impacts their risk management and profitability. This project aims to build a robust classification model that can accurately predict loan approval based on various applicant features. By leveraging machine learning algorithms, the model provides valuable insights that help in making informed lending decisions.

<img src = "https://github.com/somaksanyal97/Bank-Loan-Prediction/blob/main/Pictures/download%20(1).jpeg" style="width:1000px; height:300px;"><br>

## Data
The dataset used in this project consists of historical loan application records. Each record includes several features such as applicant income, co-applicant income, loan amount, loan amount term, credit history, gender, marital status, education, self-employment status, property area, and loan approval status. This diverse set of features is essential for capturing the factors that influence loan approval decisions.

## Preprocessing
To prepare the data for modeling, several preprocessing steps were undertaken:

* Handling Missing Values: Missing values in the dataset were imputed using appropriate techniques to ensure data completeness.
* Encoding Categorical Variables: Categorical features were encoded using one-hot encoding to transform them into a format suitable for machine learning algorithms.
* Feature Scaling: Numerical features were scaled to normalize the data, ensuring that all features contribute equally to the model.
* Data Splitting: The dataset was split into training and testing sets to evaluate the model's performance effectively.

## Modeling
Various machine learning models were tested to determine the best approach for predicting loan approval. The models evaluated include:

* Logistic Regression: A simple yet effective model for binary classification tasks.
* Decision Tree Classifier: A non-parametric model that captures non-linear relationships in the data.
* Random Forest Classifier: An ensemble learning method that improves predictive performance by combining multiple decision trees.
* XGBoost: A powerful gradient boosting algorithm known for its high accuracy and efficiency.

## Results
After extensive experimentation and model tuning, the Random Forest Classifier was selected as the final model due to its superior performance. The model achieved high accuracy on the test set, demonstrating its effectiveness in predicting loan approval status. The key metrics used to evaluate the model's performance include accuracy, precision, recall, and F1-score.

* Accuracy: The overall correctness of the model in predicting loan approval status.
* Precision: The proportion of true positive predictions among all positive predictions made by the model.
* Recall: The proportion of true positive predictions among all actual positive instances.
* F1-Score: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.

## Conclusion
This project successfully developed a robust machine learning model for predicting bank loan approval. The model's high accuracy and reliability make it a valuable tool for financial institutions, aiding in risk management and decision-making processes. Future work may involve exploring more advanced techniques and additional features to further improve the model's performance.

## Usage
To use the model, clone this repository and run the provided Jupyter notebooks. Detailed instructions for preprocessing, model training, and evaluation are included in the notebooks. The repository also includes a script for making predictions on new loan applications.

