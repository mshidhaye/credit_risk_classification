# Credit Risk Analysis Report

The purpose here is to analyze a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

ALT-H1 Overview of the Analysis

I went through the following stages of the machine learning to process the analysis: 

# A. Created label sets and feature Dataframe from the provided dataset.
First, the lending_data.csv dataset was used. This dataset contained loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, total_debt, and loan_status. The loan_status column contained either 0 or 1, where 0 meant that the loan is healthy, and 1 meant that the loan is at a high risk of defaulting. This dataset was then stored in a dataframe.
Then, I created 2 variables, x variable and y variable. The loan status column was stored in the Y variable. All the columns except loan_status were stored in the X variable. I checked the balance of the labels with value_counts. The results showed that, in our dataset, 75036 loans were healthy and 2500 were high-risk.

# B. Split the data into training and testing datasets by using train_test_split. 
I used the train_test_split module from sklearn . The data was split into training and testing variables, namely: X_train, X_test, y_train, and y_test. In order to ensure that the train/test split was consistent, I assigned a random_state of 1 to the function. The same data points are assigned to the training and testing sets across multiple runs of code.

# C. Create a logistic regression model and fit our original data into the model.
I used LogisticRegression(), from sklearn, with a random_state of 1. I fit the model with the training data, X_train and y_train, and predicted on testing data labels with predict() using the testing feature data, X_test, and the fitted model, lr_model. I calculated the accuracy score of the model with balanced_accuracy_score() from sklearn, I used y_test a d testing_prediction to obtain the accuracy.

I generated a confusion matrix for the model with confusion_matrix() from sklearn, based on y_test and testing_prediction. I obtained a classification report for the model with classification_report() from sklearn, and I used y_test and testing_prediction.


# Results
Logistic Regression Model
Model Precision: 93% (an average--in predicting low-risk loans, the model was 100% precise, though the model was only 87% precise in predicting high-risk loans)
Model Accuracy: 94%
Model Recall: 95% (an average--the model had 100% recall in predicting low-risk loans, but 89% recall in predicting high-risk loans)

# Summary

For healthy loans the precision is 1.00, the recall is 0.99, and the f1-score is 1.00, meaning that the model does really well when predicting instances. For high-risk loans the precision is 0.85, the recall is 0.91, and the f1-score is 0.88, meaning that the model does moderately well in predicting instances.

If the goal of the model was to predict likelihood of high risk loans, this model falls short(87%). I would not recommend this model due to low accuracy and recall.

