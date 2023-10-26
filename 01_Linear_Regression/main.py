from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('cells.csv')
print(df)

# plt.xlabel('time')
# plt.ylabel('cells')
# plt.scatter(df.time, df.cells,color='red',marker='+')
# plt.show()

# x_df = df.drop('cells', axis='columns')
# y_df = df.cells
x_df=df[['time']]
y_df=df[['cells']]
print(x_df)
print(y_df)

# To create a model instance

model = linear_model.LinearRegression()  # Create an instance of the model.
model.fit(x_df,y_df)   # Train the model or fits a linear model

print(model.score(x_df,y_df))  # Prints the R^2 value, a measure of how well
# observed values are replicated by the model.

# Test the model by Predicting cells for some values reg.predict()
print("Predicted # cells...", model.predict([[3.4]]))
