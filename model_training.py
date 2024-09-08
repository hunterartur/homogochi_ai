import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import pickle


# scroring function
def print_metrics(y_test, y_pred):
    '''
    Printing out the regression metrics
    :param y_test: true values of the target variable
    :param y_pred: predicted values of the target variable
    :return: calculates and prints out R2 and MAE metrics
    '''
    print(f"R2: {np.round(r2_score(y_test, y_pred), 2)} \
    \nMean_absolute_error: {np.round(mean_absolute_error(y_test, y_pred), 2)}")


# loading in the data and creating x and y variables
df = pd.read_csv('steps_prediction.csv')
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# dataframe splitting into training and test
# model fitting
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=54)
xgb = XGBRegressor(n_estimators=500, max_depth=5, random_state=54)
xgb.fit(X_train, y_train)
y_pred = (xgb.predict(X_test)).astype(int)
print_metrics(y_test, y_pred)

# model saving and loading

with open('xgb.pkl','wb') as f:
    pickle.dump(xgb, f)

# загрузка модели
with open("xgb.pkl", "rb") as f:
    model = pickle.load(f)

# checking that it works
print_metrics(y_test, (model.predict(X_test)).astype(int))

print(X_test.iloc[0,:])
a = [[1.0, 35.0, 160.0, 50.0, 3.0]]
z = model.predict(a)
print(z)
