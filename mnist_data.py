from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from numpy import atleast_2d

mnist = fetch_openml('mnist_784', as_frame=False)


X = mnist['data']
y = atleast_2d(mnist['target'].astype(int)).T

scaler = make_pipeline(MinMaxScaler((-.1, .1)), StandardScaler(with_std=False))
onehot = OneHotEncoder(categories=[range(10)], sparse_output=False)

N_train = 50_000
N_valid = 10_000
N_test  = 10_000

X_train = X[:N_train]
X_valid = X[N_train:N_train+N_valid]
X_test  = X[-N_test:]

X_train = scaler.fit_transform(X_train.reshape((-1, 1))).reshape((N_train, -1))
X_valid = scaler.transform(X_valid.reshape((-1, 1))).reshape((N_valid, -1))
X_test  = scaler.transform(X_test.reshape((-1, 1))).reshape((N_test, -1))

y_train = onehot.fit_transform(y[:N_train])
y_valid = onehot.fit_transform(y[N_train:N_train+N_valid])
y_test  = onehot.fit_transform(y[-N_test:])