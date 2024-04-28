import numpy as np    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Step 1 Load Data
dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1:].values
dataset.head()

# Step 2: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

print(dataset)

# Step 3: Fit Simple Linear Regression to Training Data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 4: Make Prediction
y_pred = regressor.predict(X_test)
regressor.score(X_test,y_test)

# save the model 
filename='final_model.sav'
joblib.dump(regressor, filename)

# Step 7 - Make new prediction
new_salary_pred = regressor.predict([[15]])
print('The predicted salary of a person with 15 years experience is ',new_salary_pred)



# Step 5 - Visualize training set results
# import matplotlib.pyplot as plt

# # plot the actual data points of training set
# plt.scatter(X_train, y_train, color = 'red')
# # plot the regression line
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

# # Step 6 - Visualize test set results
# import matplotlib.pyplot as plt
# # plot the actual data points of test set
# plt.scatter(X_test, y_test, color = 'red')
# # plot the regression line (same as above)
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience (Test set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()