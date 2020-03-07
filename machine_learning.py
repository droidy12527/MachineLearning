import numpy as np 
import pandas as pd 
import seaborn as se 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('data_housing.csv')
plt.scatter(data['price'],data['sqft_living'])
plt.title('Price vs Squarefeet')
plt.show()
y = data['price']
date_conversion = [1 if values == 2014 else 0 for values in data.date]
data['date'] = date_conversion
X = data.drop(['id' , 'price'], axis=1)
X_train , X_test , y_train, y_test = train_test_split(X,y, random_state=2,test_size=0.10 )
reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))