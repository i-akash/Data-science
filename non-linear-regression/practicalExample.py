import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import curve_fit

path="./data/non-dataset.csv"
df=pd.read_csv(path)


def eq(x,p,c):
    y = np.exp(x+p)+c
    return y

# analyzing the behaviour of data
# plt.plot(df[['ENGINESIZE']],df[['CO2EMISSIONS']],'-ro')
# plt.show()


# splitting
mask=np.random.rand(len(df))<0.7

train=df[mask]
test=df[~mask]

train_x=np.asanyarray(train['ENGINESIZE'].values)
train_y=np.asanyarray(train['CO2EMISSIONS'].values)


print(train[['ENGINESIZE']])
# analyzing training data
plt.plot(train[['ENGINESIZE']],train[['CO2EMISSIONS']],'-ro')
# plt.plot(train_x,eq(train_x,2,120),'-yo')
# plt.show()

# train 
popt,pcov=curve_fit(eq,train_x,train_y)

# test
test_x=np.asanyarray(test[['ENGINESIZE']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])

plt.plot(test_x,test_y,'-go')
plt.plot(test_x,eq(test_x,popt[0],popt[1]),'-bo')
plt.show()
