from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


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
        if self.coefs_.shape[0] != X.shape[1]:
            raise AssertionError("weight dimension is incompatible with input")

    @abstractmethod
    def score(self, X, y):
        pass

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
        self.intercept_ = np.zeros((1, y.shape[1]))
        self.coefs_ = np.zeros((X.shape[1], y.shape[1]))

        for _ in range(max_itr):
            gradient = self.cost_gradient(X, y)

            self.coefs_ -= alpha * gradient[1:, :]
            self.intercept_ -= alpha * gradient[0, :]

    def sgdescent(self, X, y, alpha=0.1, max_itr=1000, batch=1):
        raise NotImplementedError

    @property
    def weights(self) -> np.ndarray | None:
        if self.coefs_ is None or self.intercept_ is None:
            return None

        return np.vstack((self.intercept_, self.coefs_))

    @weights.setter
    def weights(self, weight_array):
        self.intercept_ = weight_array[0, :]
        self.coefs_ = weight_array[1:, :]

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
        dummies = pd.get_dummies(y, dtype="float")

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

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Sigmoid estimator function"""
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        super().predict(X)
        return sigmoid(self.intercept_ + X @ self.coefs_)

    def score(self, X, y):
        """Accuracy score"""
        pred = self.predict_class(X)
        correct_sum = (pred == y).sum()
        n_samples = X.shape[0]

        return correct_sum / n_samples

    def predict_class(self, X):
        y_pred = self.predict(X)
        y_pred = self.inv_transform_output(y_pred)

        y_pred = y_pred.set_axis(X.index)

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

        gradient_intercept = y_diff.sum(axis=1).reshape(-1, 1)
        gradient_coefs = y_diff @ X

        gradient = np.hstack((gradient_intercept, gradient_coefs))
        gradient /= X.shape[0]

        return gradient.T  # (thetas, classes)


class OneVsAllRegressor(LogisticRegressor):
    """One-vs-all logistic regression model"""
    def transform_output(self, y: pd.Series) -> np.ndarray:
        dummies = pd.get_dummies(y, dtype="float")

        classes = dummies.columns
        self.transform_info = {'name': y.name, 'classes': classes}

        return dummies.to_numpy()

    def inv_transform_output(self, y_pred: np.ndarray) -> pd.Series:
        if self.transform_info is None:
            raise AssertionError("tranformation info is unavailable")

        classes = self.transform_info["classes"]
        df = pd.DataFrame(y_pred, columns=classes)

        return df.idxmax(axis=1).rename(self.transform_info["name"])

    def predict_tabular(self, X) -> pd.DataFrame:
        indices = X.index
        classes = self.transform_info["classes"]

        y_pred = self.predict(X)

        return pd.DataFrame(y_pred,
                            index=indices,
                            columns=classes)

    def load_weights(self, weights_file, label="class"):
        """Load weights from a csv file"""
        df = pd.read_csv(weights_file, index_col=0)
        self.weights = df.to_numpy()
        self.transform_info = {}
        self.transform_info["name"] = label
        self.transform_info["classes"] = df.columns


def sigmoid(z):
    return 1. / (1. + np.exp(-z))
