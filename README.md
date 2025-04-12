# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: K.HEMANATH
RegisterNumber: 212223100012
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("Employee.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

print(data["left"].value_counts())

le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
y = data["left"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print("Prediction for input [0.5, 0.8, 9, 260, 6, 0, 1, 2]:", prediction)

plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=x.columns, class_names=['Not Left', 'Left'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
```

## Output:
![1](https://github.com/user-attachments/assets/f38fe070-ea1f-409d-ae75-fa99af4dd4d1)

![2](https://github.com/user-attachments/assets/50a09662-56f5-4530-9f6a-eca348c23701)

![3](https://github.com/user-attachments/assets/159ba0cc-1c4a-4d88-af3d-c16169c1d181)

![4](https://github.com/user-attachments/assets/42fe950c-6ed8-4327-98b2-71915d54e18c)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
