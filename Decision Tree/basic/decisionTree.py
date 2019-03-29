import pandas as pd
import numpy as np
from  sklearn import preprocessing as prepro
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

df=pd.read_csv('./data/drug200.csv')



# sigficant attribute
X=df[['Age','Sex','BP','Cholesterol', 'Na_to_K']].values

# normalize non numerical value
def convertToNumerical(col,fittedLabel):
    label=prepro.LabelEncoder()
    label.fit(fittedLabel)
    return label.transform(X[:,col])


X[:,1]=convertToNumerical(1,['M','F'])
X[:,2]=convertToNumerical(2,['LOW','HIGH','NORMAL'])
X[:,3]=convertToNumerical(3,['LOW','HIGH','NORMAL'])


# target atribute
Y=df['Drug'].values

# splitting
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=3)


# decision tree
drugTree=DecisionTreeClassifier(criterion='entropy',max_depth=4)
drugTree.fit(x_train,y_train)


# prediction
y_predict=drugTree.predict(x_test)

# score
print('accuracy : ',metrics.accuracy_score(y_test,y_predict))

print("-"*80)
print(y_predict)
print("-"*80)
print(y_test)



