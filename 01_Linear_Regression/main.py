from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("cells.csv")
print(df)

# sns.set(style='darkgrid')
# sns.lmplot(x='time', y='cells', data=df, order=1)

plt.xlabel('time')
plt.ylabel('cells')
plt.scatter(df.time, df.cells,color='red',marker='*')
plt.show()

# Y = dependent variable = the value we want to predict
# X = independent variables upon which Y depends
# 3 steps for linear regression....
# Step 1: Create the instance of the model
# Step 2: .fit() to train the model or fit a linear model
# Step 3: .predict() to predict Y for given X values.

# x_df = df.drop('cells', axis='columns')
# y_df = df.cells
x_df=df[['time']]
y_df=df[['cells']]
print(x_df)
print(y_df)

# To create a model instance

# Step 1 : Create an instance of the model
model = linear_model.LinearRegression()

# Step 2 : Train the model or fits a linear model
model.fit(x_df, y_df)

print(model.score(x_df, y_df))  # Prints the R^2 value, a measure of how well
# observed values are replicated by the model.

# Step 3 : Test the model by Predicting cells for some values model.predict()
print("Predicted number of cells = ", model.predict([[3.4]]))

# Y = m * X + b (m is coefficient and b is intercept)
# Get the intercept and coefficient values
b = model.intercept_
m = model.coef_

# Manually verify the above calculation
print("From manual calculation, cells = ", (m*3.4 + b))

# Step 3 : Now predict cells for a list of times by reading time values from a csv file
cells_predict_df = pd.read_csv("test_cells.csv")
print(cells_predict_df.head())

predicted_cells = model.predict(cells_predict_df)
print(predicted_cells)

# Add the new predicted cells values as a new column to cells_predict_df dataframe
cells_predict_df['cells'] = predicted_cells
print(cells_predict_df)

cells_predict_df.to_csv("predicted_cells.csv")