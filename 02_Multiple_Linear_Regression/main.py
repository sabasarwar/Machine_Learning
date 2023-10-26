import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Reading the excel file
df = pd.read_excel("images_analyzed.xlsx")
print(df.head())

# A few plots in Seaborn to understand the data
sns.lmplot(x='Time', y='Images_Analyzed', data=df, hue='Age')
# Scatterplot with linear regression fit and 95% confidence interval
sns.lmplot(x='Coffee', y='Images_Analyzed', data=df, hue='Age', order=2)
# Looks like too much coffee is not good... negative effects
plt.show()

# Create Linear Regression object
model = linear_model.LinearRegression()

# Call fit method to train the model using independent variables.
# And the value that needs to be predicted (Images_Analyzed)
# Independent variables = x = df[['Time', 'Coffee', 'Age']]
# Dependent variable to be predicted = y = df.Images_Analyzed
# model.fit(df[['Time', 'Coffee', 'Age']], df.Images_Analyzed)

# To avoid warning use only values of the dataframe
model.fit(df[['Time', 'Coffee', 'Age']].values, df.Images_Analyzed.values)

# Model is ready. Let us check the coefficients, stored as reg.coef_
# These are a, b, and c from our equation.
# Intercept is stored as reg.intercept_
print("\nCoefficients are : ", model.coef_)
print("\nIntercept is : ", model.intercept_)

# All set to predict the number of images someone would analyze at a given time
print("\nThe predicted value (Number of Images) for given Time=13, Coffee=2, Age=23 are : ")
print("\nNumber of Images : ", model.predict([[13, 2, 23]]))

