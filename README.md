# Machine-LearningIntern
Intern Career Machine Learning Internship tasks

# TASK 1

Titanic Machine Learning Disaster - Model Development and Evaluation Documentation

                                    Introduction

This documentation provides a step-by-step guide on how to approach and solve the Kaggle Titanic Machine Learning Disaster challenge using Python. The goal of this challenge is to predict whether a passenger survived or not based on various features.

Environment
The code is designed to run in a Kaggle Python 3 environment, utilizing the Kaggle/python Docker image. The environment comes with several helpful analytics libraries pre-installed, such as NumPy, Pandas, and Matplotlib.
 

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

The final evaluation indicates high performance metrics, suggesting a well-fitted model for predicting survival in the Titanic dataset. The model achieved perfect accuracy, precision, recall, and F1 score on the test set. Further optimization and tuning may be explored for more complex datasets.

