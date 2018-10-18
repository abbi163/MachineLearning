import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("E:/Pythoncode/Coursera/CO2 Emission/FuelConsumption.csv")

cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB","CO2EMISSIONS"]]

print(cdf.head())
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

# splitting the dataset into mutually exclusive test and train data set

msk = np.random.rand(len(df))<0.8

train = cdf[msk]
test = cdf[~msk]

# Multiple Regression on train data set

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print ('Coefficients:', regr.coef_)
print ('Intercept:', regr.intercept_)

test_y_hat = regr.predict(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
test_x = np. asanyarray(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])
print('Residual sum of squares: %s' % np.mean((test_y_hat - test_y)**2))

# Explained variance score: 1 is perfect predicition
print("Variance Score: %s" % regr.score(test_x,test_y))
