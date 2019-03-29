import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# read
df=pd.read_csv('./data/teleCust1000t.csv')
X=df[['region','tenure','age','marital','address','income','ed','employ','retire','gender','reside']].values

Y=df['custcat'].values

# normalize 
X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))


# split
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=.2,random_state=4)


k=3

# train model
neigh=KNeighborsClassifier(n_neighbors=k).fit(train_x,train_y)


# predicting
predict_y=neigh.predict(test_x)


# accuracy

print("train : ",metrics.accuracy_score(train_y,neigh.predict(train_x)))
print("train : ",metrics.accuracy_score(test_y,neigh.predict(test_x)))



# df.hist(column='income',bins=50)
# plt.show()