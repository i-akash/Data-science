
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
#read data

path="./data/Dataset.csv"
df=pd.read_csv(path)

#ploting data or relation on graph

x=np.asanyarray(df[["ENGINESIZE"]])
y=np.asanyarray(df[["CO2EMISSIONS"]])

# plt.scatter(x,y,color="red")
# plt.xlabel("engine")
# plt.ylabel("emission")
# plt.show()


# splitting data train and test
mask=np.random.rand(len(df))<0.8
train=df[mask]
test=df[~mask]

print("mask :",mask)
# train

regr=linear_model.LinearRegression()

train_x=np.asanyarray(train[["ENGINESIZE","CYLINDERS"]])
train_y=np.asanyarray(train[["CO2EMISSIONS"]])

regr.fit(train_x,train_y)

print("co-ef :",regr.coef_)
print("intercept :",regr.intercept_)


# showing regression line over train data
# plt.scatter(train_x,train_y,color='blue')
# plt.plot(train_x,regr.coef_[0][0]*train_x[0][0]+regr.coef_[0][1]*train_x[0]+regr.intercept_[0],'-r')



#predict 
test_x=np.asanyarray(test[["ENGINESIZE","CYLINDERS"]])
test_y=np.asanyarray(test[["CO2EMISSIONS"]])

predict_y=regr.predict(test_x)

# plt.scatter(test_x,test_y,color="green")
# plt.scatter(test_x,predict_y,color="brown")
plt.show()

print("absolute mean :",np.mean(np.absolute(predict_y-test_y)))
print("score :",r2_score(predict_y,test_y))
print("test x :",test_y,"predict y:",predict_y)
