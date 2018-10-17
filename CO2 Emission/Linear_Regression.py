import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

df = pd.read_csv("E:/Pythoncode/Coursera/CO2 Emission/FuelConsumption.csv")

cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]

# splitting the data set into training(80%) and test(20%) set.

msk = np.random.rand(len(df))<0.80
train = cdf[msk]
test = cdf[~msk]

# train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color ="blue")
plt.title("Train Data Set")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

# Linear Regression on train data set
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print ('Coefficient:', regr.coef_)
print ('Intercept:', regr.intercept_)


# plotting the best fit line over the data

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color ="blue")
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# predicting test data and finding r2_score
from sklearn.metrics import r2_score

test_x = np.asanyarray(train[['ENGINESIZE']])
test_y = np.asanyarray(train[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print(" Mean absolute error : %s" % np.mean(np.absolute(test_y_hat - test_y)))
print ("Residual sum of square (MSE): %s" % + np.mean((test_y_hat - test_y)**2))
print ("R2-score:%s" % + r2_score(test_y_hat ,test_y))
