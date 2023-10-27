import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# STEP 1: DATA READING AND UNDERSTANDING
df = pd.read_csv("images_analyzed_productivity.csv")
print(df.head())

# plt.scatter(df.Age, df.Productivity, marker='+', color='red')
# plt.show()
# plt.scatter(df.Time, df.Productivity, marker='+', color='red')
# plt.show()
# plt.scatter(df.Coffee, df.Productivity, marker='+', color='red')
# plt.show()

# PLot productivity values to see the split between Good and Bad
sizes = df['Productivity'].value_counts(sort=1)

# Pie plot
# plt.pie(sizes, shadow=True, autopct='%1.1f%%')
# plt.show()

# STEP 2: DROP IRRELEVANT DATA
# Images_Analyzed reflects whether it is good analysis or bad
# so should not include it. ALso, User number is just a number and has no inflence
# on the productivity, so we can drop it.

df.drop(['Images_Analyzed'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)
print(df.head())

# STEP 3: Handle missing values, if needed
# df = df.dropna()  #Drops all rows with at least one null value.


# STEP 4: Convert non-numeric to numeric, if needed.
# Sometimes we may have non-numeric data, for example batch name, user name, city name, etc.
# e.g. if data is in the form of YES and NO then convert to 1 and 2

df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 2
# print(df.head())

# STEP 5: PREPARE THE DATA.

# Y is the data with dependent variable, this is the Productivity column
Y = df["Productivity"].values  # At this point Y is an object not of type int
# Convert Y to int
Y = Y.astype('int')

# X is data with independent variables, everything except Productivity column
# Drop label column from X as you don't want that included as one of the features
X = df.drop(labels=["Productivity"], axis=1)
# print(X.head())

# STEP 6: SPLIT THE DATA into TRAIN AND TEST data
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.4, random_state=20)

# random_state can be any integer, it is used as a seed to randomly split dataset.
# By doing this we work with same test dataset every time, if this is important.
# random_state=None splits dataset randomly every time

# print(train_X)

# STEP 7: Defining the model and training.
model = LogisticRegression()  # Create an instance of the model.

model.fit(train_X, train_y)  # Train the model using training data

# STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA AND CALCULATE THE ACCURACY SCORE
prediction_test = model.predict(test_X)

# Print the prediction accuracy
# Test accuracy for various test sizes and see how it gets better with more training data
print("\nAccuracy = ", metrics.accuracy_score(test_y, prediction_test))

# UNDERSTAND WHICH VARIABLES HAVE MOST INFLUENCE ON THE OUTCOME
# To get the weights of all the variables

print(model.coef_) # Print the coefficients for each independent variable.
# But it is not clear which one corresponds to what.
# SO let us print both column values and coefficients.
# .Series is a 1-D labeled array capable of holding any data type.
# Default index would be 0,1,2,3... but let us overwrite them with column names for X (independent variables)
weights = pd.Series(model.coef_[0], index=X.columns.values)

print("\nWeights for each variables is a follows : ")
print(weights)

# +VE VALUE INDICATES THAT THE VARIABLE HAS A POSITIVE IMPACT


