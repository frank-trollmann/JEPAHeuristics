import joblib
from sklearn.linear_model import LinearRegression


def train_linear_regression(X_train, y_train):
  model = LinearRegression()
  model.fit(X_train, y_train)
  return model


def save_linear_regression(model, path):
  joblib.dump(model, path)


def load_linear_regression(path):
  return joblib.load(path)
