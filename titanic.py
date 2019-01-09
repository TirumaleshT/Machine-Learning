import pandas as pd
from sklearn import linear_model, model_selection, preprocessing, svm
import matplotlib.pyplot as plt

data = pd.read_csv('titanic_data.csv', delimiter=',')

label_enc = preprocessing.LabelEncoder()
data['Sex'] = label_enc.fit_transform(data['Sex']).astype('str')

data.drop(columns=['PassengerId','Fare','Cabin','Name','Ticket','Embarked'], inplace=True)
data.dropna(inplace=True)

X = data.drop(columns=['Survived'])
y = data[['Survived']]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)

log_regr = linear_model.LogisticRegression()
log_regr.fit(X_train, y_train)

clf = svm.SVC(gamma='auto')
clf.fit(X_train, y_train)

y_pred = log_regr.predict(X_test)
print(log_regr.score(X_test, y_test))
print(clf.score(X_test, y_test))


