from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# Example TO DO:
# add transformer class in another file


class RegressionModel(ABC):
    """Template regression model class"""
    def __init__(self):
        self.intercept_ = None
        self.coefs_ = None

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the outcome from features in X

        ABC checks for errors
        """
        if self.coefs_ is None or self.intercept_ is None:
            raise AssertionError("weights are not trained")
        if self.intercept_.shape[0] != X.shape[1]:
            raise AssertionError("weight dimension is incompatible with input")

    def fit(self, X, y, **kwargs):
        """Perform regression algorithm"""
        if kwargs.get('batch') is None:
            self.gdescent(X, y, **kwargs)
        else:
            self.sgdescent(X, y, **kwargs)

    @abstractmethod
    def cost(self, X, y):
        pass

    @abstractmethod
    def cost_gradient(self, X, y):
        pass

    def gdescent(self, X, y, alpha=0.1, max_itr=1000):
        raise NotImplementedError

    def sgdescent(self, X, y, alpha=0.1, max_itr=1000, batch=1):
        raise NotImplementedError

    @property
    def weights(self) -> np.ndarray | None:
        if self.coefs_ is None or self.intercept_ is None:
            return None

        return np.vstack((self.intercept_, self.coefs_))

    def __str__(self):
        return str(self.weights)

    def __repr__(self):
        return repr(self.weights)


class LogisticRegressor(RegressionModel):
    """Binary logistic regression model"""
    def __init__(self):
        super().__init__()
        self.transform_info = None

    def transform_output(self, y: pd.Series) -> np.ndarray:
        dummies = pd.get_dummies(y, dtype=pd.Float32Dtype)

        classes = dummies.columns
        self.transform_info = {'name': y.name, 'classes': classes}

        return dummies.iloc[:, 0].to_numpy()

    def inv_transform_output(self, y_pred: np.ndarray) -> pd.Series:
        if self.transform_info is None:
            raise AssertionError("tranformation info is unavailable")

        classes = self.transform_info["classes"]
        estimates = [classes[0] if x > .5 else classes[1] for x in y_pred]

        return pd.Series(estimates,
                         name=self.transform_info["name"])

    def fit(self, X, y, **kwargs):
        X = X.to_numpy()
        y = self.transform_output(y)

        super().fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid estimator function"""
        super().predict(X)
        return sigmoid(self.intercept_ + X @ self.coefs_)

    def predict_class(self, X):
        indices = X.index

        y_pred = self.predict(X)
        y_pred = self.inv_transform_output(y_pred)

        y_pred.rename(index=indices, inplace=True)

        return y_pred

    def cost(self, X, y):
        """Log-loss function"""
        y_pred = self.predict(X)

        to_change = y == 1.
        y_pred[to_change] = 1. - y_pred[to_change]

        loss_sum = np.log(y_pred).sum(axis=0)

        return -(loss_sum / y.shape[0])

    def cost_gradient(self, X, y):
        """Log-loss function for sigmoid estimator"""
        y_diff = self.predict(X) - y
        y_diff = y_diff.T  # class in row(s), samples in columns

        gradient_intercept = y_diff.sum(axis=1)
        gradient_coefs = y_diff @ X

        gradient = np.hstack((gradient_intercept, gradient_coefs))
        gradient /= X.shape[0]

        return gradient.T  # (thetas, classes)


class OneVsAllRegressor(LogisticRegressor):
    """One-vs-all logistic regression model"""
    def transform_output(self, y: pd.Series) -> np.ndarray:
        dummies = pd.get_dummies(y, dtype=pd.Float32Dtype)

        classes = dummies.columns
        self.transform_info = {'name': y.name, 'classes': classes}

        return dummies.to_numpy()

    def inv_transform_output(self, y_pred: np.ndarray) -> pd.Series:
        if self.transform_info is None:
            raise AssertionError("tranformation info is unavailable")

        classes = self.transform_info["classes"]
        df = pd.DataFrame(y_pred, columns=pd.Index(classes))

        return df.idxmax(axis=1).rename(self.transform_info["name"])

    def predict_tabular(self, X) -> pd.DataFrame:
        indices = X.index
        classes = self.transform_info["classes"]

        y_pred = self.predict(X)
        y_pred = self.inv_transform_output(y_pred)

        y_pred.rename(index=indices, inplace=True)

        return pd.DataFrame(y_pred,
                            index=indices,
                            columns=pd.Index(classes))


def sigmoid(z):
    return 1. / (1. + np.exp(-z))
