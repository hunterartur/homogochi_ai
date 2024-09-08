import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import KBinsDiscretizer, minmax_scale, binarize

# creaating simulated data
X, y = make_regression(n_samples=5000, n_features=5, n_informative=5, random_state=54)

# adjusting it to look natural: renaming columns, binarizing and min_max_scaling
x = pd.DataFrame(X)
y = pd.Series(y)
y = pd.Series(minmax_scale(y, feature_range = (1000, 15000))).astype(int)

col_names = ['gender', 'age', 'height', 'weight', 'daily_activity']
print(x.head())
x.rename(columns=dict(zip(list(x.columns), col_names)), inplace = True)
print(x.head())

x['gender'] = binarize(x.iloc[:,0:1])
x['age'] = minmax_scale(x.iloc[:,1:2], feature_range = (15, 99))
x['height'] = minmax_scale(x.iloc[:,2:3], feature_range = (130, 200))
x['weight'] = minmax_scale(x.iloc[:,3:4], feature_range = (30, 150))
kbinizer = KBinsDiscretizer(n_bins=5, encode='ordinal', dtype=None, subsample=1000, random_state=None)
x['daily_activity'] = kbinizer.fit_transform(x.iloc[:,4:5])
x = x.round()
x.head()

# saving it locally as a .csv file
df = x.copy()
df['steps'] = y
df.head()
df.to_csv('steps_prediction.csv', index=False)