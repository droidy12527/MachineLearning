import numpy as np 
import pandas as pd 
import seaborn as se 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import ensemble
data = pd.read_csv('data_housing.csv')
ans = input('Do you want to see graph for the dataset ?')
if ans.lower() == 'yes':
    plt.scatter(data['price'],data['sqft_living'])
    plt.title('Price vs Squarefeet')
    plt.show()
y = data['price']
date_conversion = [1 if values == 2014 else 0 for values in data.date]
data['date'] = date_conversion
X = data.drop(['id' , 'price'], axis=1)
X_train , X_test , y_train, y_test = train_test_split(X,y, random_state=2,test_size=0.10 )
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,learning_rate = 0.1, loss = 'ls')
clf.fit(X_train, y_train)
print('Accuracy = ' + str(clf.score(X_test,y_test)*100))

