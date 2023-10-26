from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("cells.csv")
print(df)

# Y = dependent variable = the value we want to predict
# X = independent variables upon which Y depends
# 3 steps for linear regression....
# Step 1: Create the instance of the model
# Step 2: .fit() to train the model or fit a linear model
# Step 3: .predict() to predict Y for given X values.

# x_df = df.drop('cells', axis='columns')
# y_df = df.cells
x_df = df[['time']]
y_df = df[['cells']]
print(x_df)
print(y_df)

# random_state can be any integer and it is used as a seed to randomly split dataset.
# By doing this we work with same test dataset every time, if this is important.
# random_state = None => splits dataset randomly every time

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(x_df, y_df, test_size=0.4, random_state=10)


# Step 1 : Create an instance of the model
model = linear_model.LinearRegression()
# Step 2 : Train the model or fits a linear model
model.fit(train_X, train_y)

print(model.score(train_X, train_y))  # Prints the R^2 value, a measure of how well
# observed values are replicated by the model

# Step 3 : Test the model by Predicting cells for some values model.predict()
predicted_test = model.predict(test_X)
print(test_y, predicted_test)
print("Mean square error between test_y and predicted = ", np.mean((predicted_test-test_y)**2))
# A MSE value of about 40 is not bad compared to average # cells about 250.

# Residual plot
plt.scatter(predicted_test, predicted_test-test_y)
plt.hlines(y=0, xmin=200, xmax=300)
plt.show()



