import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
df=pd.read_csv('teleCust1000t.csv')
print(df.tail(10))
print(df.shape)
print(df['custcat'].value_counts())
print(df.columns)
x=df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside']].values
y=df['custcat'].values
x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))
print('x:', x[:,5])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print('Train set',x_train.shape,y_train.shape)
print('Test set',x_test.shape,y_test.shape)
#Classification
from sklearn.neighbors import KNeighborsClassifier
K=4
neigh=KNeighborsClassifier(n_neighbors=K).fit(x_train,y_train)
print(neigh)
#Predicting
yhat=neigh.predict(x_test)
print(yhat[0:5])
print(y[0:5])
#Accuracy evaluation
from sklearn import metrics
print('Test set Accuaracy:' , metrics.accuracy_score(y_test,yhat))
print('Train set Accuaracy:' , metrics.accuracy_score(y_train,neigh.predict(x_train)))
# Found the optimal value of K
Ks=10
mean_acc=np.zeros((Ks-1))
std_acc=np.zeros((Ks-1))
for n in range(1,Ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1]=metrics.accuracy_score(y_test,yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
print(mean_acc)
print('The best accuracy was with', mean_acc.max, 'with K=', mean_acc.argmax()+1)
nx=[[2,13,44,1,9,1500,4,5,0,0,2]]
scaler=preprocessing.StandardScaler().fit(x)
nxs=scaler.transform(nx)
print(neigh.predict(nxs))
