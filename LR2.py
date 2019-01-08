# importing dependancis
import numpy as np
import pandas as pd
from sklearn import model_selection, linear_model, preprocessing
import matplotlib.pyplot as plt
import datetime as dt

# Load data and prepare data for modelling

columns = ['device','humidity','temperature','experiment','time']
dataset = pd.read_csv('gnfuv-temp-exp1-55d487b85b-5g2xh_1.0.csv', delimiter=',', names=columns)           

def converter(df, column):
    empty = []
    for i in range(len(df)):
        lis = list(df.loc[:, column])
        a = str(lis[i]).split(':')
        empty.append(float(a[-1]))
    return empty

def converter1(df, column):
    empty = []
    for i in range(len(df)):
        lis = list(df.loc[:, column])
        a = str(lis[i]).split(':')
        s = str(a[-1]).split('}')
        empty.append(float(s[0]))
    return empty

humidity = converter(dataset, 'humidity')
temperature = converter(dataset, 'temperature')
time = converter1(dataset, 'time')

#Unix time conversion
timeconver = np.vectorize(dt.datetime.fromtimestamp)
time_new = timeconver(time)

preprocessed_data = pd.DataFrame({'humidity':humidity, 'temperature':temperature, 'time':time_new})

#Visualize data
plt.scatter(humidity, temperature, label='time - humidity')
#plt.plot(time_new, temperature, label='time - temperature')
plt.legend()
plt.show()

X = preprocessed_data[['humidity']]
y = preprocessed_data[['temperature']]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.3)

#modelling
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test,y_test))











