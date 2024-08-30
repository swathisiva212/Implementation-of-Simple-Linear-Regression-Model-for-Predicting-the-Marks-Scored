# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.   

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: swathi.s
RegisterNumber:  212223040219.
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
````

## Output:

![image](https://github.com/user-attachments/assets/6ed8069b-3a3c-4d64-8467-37b87f8e7553)

![image](https://github.com/user-attachments/assets/503fc102-aebc-4afb-a4c1-a8904a6824b5)
![image](https://github.com/user-attachments/assets/f5479ed9-3361-4c59-b874-bab96b46a27a)
![image](https://github.com/user-attachments/assets/0271b681-8dd7-4ed0-851e-6a68a7cb6b27)
![image](https://github.com/user-attachments/assets/b9c2c19b-6528-47e7-ad91-f64d9decdf33)
![image](https://github.com/user-attachments/assets/9c9659f6-fbff-4abf-99e8-0160e990a3be)
![image](https://github.com/user-attachments/assets/c44eda38-b50d-4189-b9e3-2837849a3b9b)
![image](https://github.com/user-attachments/assets/6b3b804e-048d-4fb7-976a-2eb5dd87cb32)
![image](https://github.com/user-attachments/assets/b2ddf217-fda3-4925-8a88-81c246e53d7c)
![image](https://github.com/user-attachments/assets/37e941d4-8e51-4546-8a64-61d2438f5440)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
