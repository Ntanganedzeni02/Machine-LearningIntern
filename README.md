# Machine-LearningIntern
Intern Career Machine Learning Internship tasks

# TASK 1

Titanic Machine Learning Disaster - Model Development and Evaluation Documentation

                                    Introduction

This documentation provides a step-by-step guide on how to approach and solve the Kaggle Titanic Machine Learning Disaster challenge using Python. This challenge aims to predict whether a passenger survived or not based on various features.

Environment
The code is designed to run in a Kaggle Python 3 environment, utilizing the Kaggle/python Docker image. The environment has several helpful analytics libraries pre-installed, such as NumPy, Pandas, and Matplotlib.
 

                                    Data Loading

The dataset consists of three CSV files –“gender_submission.csv”, "train.csv" and "test.csv," located in the "/kaggle/input/titanic-machine-earning-disaster/" directory. These files contain information about passengers, including features like PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, and Embarked.

The code reads the training and test datasets into Pandas DataFrames:
train_data = pd.read_csv("/kaggle/input/titanic-machine-earning-disaster/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic-machine-earning-disaster/test.csv")

                                  Exploratory Data Analysis
The initial exploration includes displaying the first few rows of the training and test datasets using the `head()` method.

train_data.head()
test_data.head()
 

Survival Rate Analysis

The code calculates and prints the percentage of women and men who survived from the training dataset:
rate_women = sum(train_data.loc[train_data.Sex == 'female']["Survived"]) / len(train_data.loc[train_data.Sex == 'female'])
rate_men = sum(train_data.loc[train_data.Sex == 'male']["Survived"]) / len(train_data.loc[train_data.Sex == 'male'])

print("% of women who survived:", rate_women)
print("% of men who survived:", rate_men)
 

                                      Model Training

A Random Forest Classifier is used to build the predictive model. The model is trained on selected features ('Pclass', 'Sex', 'SibSp', 'Parch'):

from sklearn.ensemble import RandomForestClassifier
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
 

                                        Results Saving

The predictions are saved to a CSV file named "Results.csv":

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('Results.csv', index=False)
print("Your Results file was successfully saved!")

                                        Model Evaluation

The code evaluates the model's performance using various metrics such as accuracy, precision, recall, F1 score, and confusion matrix:

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

*Load the ground truth labels for the test set* :

ground_truth = pd.read_csv("/kaggle/working/Results.csv")
y_true = ground_truth['Survived']

*Evaluate the performance of the model* :

accuracy = accuracy_score(y_true, predictions)
precision = precision_score(y_true, predictions)
recall = recall_score(y_true, predictions)
f1 = f1_score(y_true, predictions)
conf_matrix = confusion_matrix(y_true, predictions)

*Display the evaluation metrics* :

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

*Display the confusion matrix*:

print("Confusion Matrix:")
print(conf_matrix)
 

                                             Conclusion

The final evaluation indicates high-performance metrics, suggesting a well-fitted model for predicting survival in the Titanic dataset. The model achieved perfect accuracy, precision, recall, and F1 score on the test set. Further optimization and tuning may be explored for more complex datasets.


# TASK 2

                                             Introduction

This documentation outlines the steps taken to analyze the Breast Cancer Wisconsin dataset. The goal is to build and evaluate machine learning models to predict the presence of malignant (M) or benign (B) tumors based on various features. The Breast Cancer Wisconsin dataset contains information about various features extracted from digitized images of breast cancer biopsies. The target variable is the diagnosis (Malignant: 1, Benign: 0).
- Total Instances: 569
- Features: 30 numerical features representing mean, standard deviation, and worst values for different characteristics.
- Target Variable: 'diagnosis' (M/B)

                                              Data Loading 

The dataset was loaded using the `pandas` library, and basic information about the dataset was displayed.

import pandas as pd

# Load the dataset
wisconsin_data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin/data.csv")

                                              Data Preprocessing

The dataset was preprocessed by removing unnecessary columns ('Unnamed: 32' and 'id'). The 'diagnosis' column was converted to binary labels (Malignant: 1, Benign: 0). The data was split into features (X) and the target variable (y).

*Drop unnecessary columns*

wisconsin_data.drop(['Unnamed: 32','id'], axis=1, inplace=True)

*Convert 'diagnosis' to binary labels*

wisconsin_data.diagnosis = [1 if each == "M" else 0 for each in wisconsin_data.diagnosis]

*Separate features and target variable*

X = wisconsin_data.loc[:, wisconsin_data.columns != "diagnosis"]
y = wisconsin_data.loc[:, "diagnosis"]
4. Model Training <a name="model-training"></a>

                                          Logistic Regression 
The logistic regression model was trained using the `LogisticRegression` class from `scikit-learn`.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

*Split the dataset into training and testing sets*

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

*Standardize the features*

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

*Create and train the Logistic Regression model*

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

*Make predictions on the test set*

y_pred = model.predict(X_test_scaled)

*Evaluate the model*

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)


                                            Decision Tree Classifier 

A decision tree classifier was trained using the `DecisionTreeClassifier` class.
from sklearn.tree import DecisionTreeClassifier

*Create and train the Decision Tree Classifier*

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

*Make predictions on the training and test sets*

train_predictions = classifier.predict(X_train)
test_predictions = classifier.predict(X_test)

                                        Model Evaluation 

Both models were evaluated using accuracy, confusion matrix, and f1-score.

*Logistic Regression Evaluation*

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report_str)

*Decision Tree Classifier Evaluation*

print("\nDecision Tree Classifier:")
print("Train Confusion Matrix:")
print(confusion_matrix(y_train, train_predictions))
print("Train Accuracy:", accuracy_score(y_train, train_predictions))
print("Train f1-score for class '1':", f1_score(y_train, train_predictions, pos_label=1))
print("Train f1-score for class '0':", f1_score(y_train, train_predictions, pos_label=0))

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))
print("Test Accuracy:", accuracy_score(y_test, test_predictions))
print("Test f1-score for class '1':", f1_score(y_test, test_predictions, pos_label=1))
print("Test f1-score for class '0':", f1_score(y_test, test_predictions, pos_label=0))

                                             Conclusion 

The analysis demonstrates the successful training and evaluation of logistic regression and decision tree classifier models on the Breast Cancer Wisconsin dataset. The models exhibit high accuracy and provide valuable insights into predicting malignant and benign tumors.


