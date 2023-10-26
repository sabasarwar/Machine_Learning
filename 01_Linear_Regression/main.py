from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("cells.csv")
print(df)

plt.xlabel('time')
plt.ylabel('cells')
plt.scatter(df.time, df.cells,color='red',marker='*')
plt.show()

# x_df = df.drop('cells', axis='columns')
# y_df = df.cells
x_df=df[['time']]
y_df=df[['cells']]
print(x_df)
print(y_df)

# To create a model instance

model = linear_model.LinearRegression()  # Create an instance of the model.
model.fit(x_df, y_df)   # Train the model or fits a linear model

print(model.score(x_df, y_df))  # Prints the R^2 value, a measure of how well
# observed values are replicated by the model.

# Test the model by Predicting cells for some values reg.predict()
print("Predicted number of cells = ", model.predict([[3.4]]))
# Y = m * X + b (m is coefficient and b is intercept)
# Get the intercept and coefficient values

b = model.intercept_
m = model.coef_

# Manually verify the above calculation
print("From manual calculation, cells = ", (m*3.4 + b))

# Now predict cells for a list of times by reading time values from a csv file
cells_predict_df = pd.read_csv("test_cells.csv")
print(cells_predict_df.head())

predicted_cells = model.predict(cells_predict_df)
print(predicted_cells)

# Add the new predicted cells values as a new column to cells_predict_df dataframe
cells_predict_df['cells'] = predicted_cells
print(cells_predict_df)

cells_predict_df.to_csv("predicted_cells.csv")