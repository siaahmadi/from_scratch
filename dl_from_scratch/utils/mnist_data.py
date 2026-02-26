import numpy as np

from sklearn.datasets import fetch_openml # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler # pyright: ignore[reportMissingModuleSource]
from sklearn.pipeline import make_pipeline # pyright: ignore[reportMissingModuleSource]

### Prepare data
mnist = fetch_openml('mnist_784', as_frame=False)

scaler = make_pipeline(MinMaxScaler(feature_range=(-.25, .25)), StandardScaler(with_std=False))
onehot = OneHotEncoder(categories=[np.arange(10)], sparse_output=False)

X = mnist['data']
y = onehot.fit_transform(mnist['target'].astype(int)[:, np.newaxis])

X_train, y_train = scaler.fit_transform(X[:50_000])  , y[:50_000]
X_valid, y_valid = scaler.transform(X[50_000:60_000]), y[50_000:60_000]
X_test,  y_test  = scaler.transform(X[60_000:])      , y[60_000:]
