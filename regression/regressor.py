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
        self.transform_info = None

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the outcome from features in X"""
        if self.coefs_ is None or self.intercept_ is None:
            raise AssertionError("weights are not trained")

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Manage input/output variables and perform regression algorithm"""
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
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid estimator function"""
        return sigmoid(self.intercept_ + self.coefs_ @ X)

    def predict_class(self, X):
        raise NotImplementedError

    def cost(self, X, y):
        """Log-loss function"""
        y_pred = self.predict(X)

        to_change = y_pred == 1.
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

        return gradient


class OneVsAllRegressor(LogisticRegressor):
    """One-vs-all logistic regression model"""

    # TODO: Restruct cost function in LogisticRegressor
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_tabular(self, X) -> pd.DataFrame:
        raise NotImplementedError

    def predict_class(self, X: np.ndarray) -> str:
        raise NotImplementedError


def sigmoid(z):
    return 1. / (1. + np.exp(-z))
