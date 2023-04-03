import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import style
from datetime import datetime

import sklearn.model_selection
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("Refined_TK_data.csv",sep=",")
print(data.head())

data = data[["Assessed Value","Retention Cost",""]]

predict = "Assessed Value"

X = np.array(data.drop([predict],1))
Y = np.array(data[predict])
best = 0
for _ in range(30):

    x_train, x_test, y_train,  y_test = sklearn.model_selection.train_test_split(X,Y,test_size = 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        with open("NewModel.pickle","wb") as f:
            pickle.dump(linear,f)

pickle_in = open("NewModel.pickle","rb")
linear = pickle.load(pickle_in)

print("Co: \n", linear.coef_)
print("Intercept",linear.intercept_)
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "Assessed Value"
style.use("ggplot")
pyplot.scatter(data[p], data["Assessed Value"])

pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()