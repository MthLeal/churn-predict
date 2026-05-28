import numpy as np

class LogisticRegressionScratch():
    def __init__(self, learning_rate=0.001, iterations=1000, _lambda=0):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.iterations = iterations
        self._lambda = _lambda


    def _sigmoid(self, z: float | np.ndarray):
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid
    

    def _loss(self, y: np.ndarray | int, y_pred: np.ndarray | int):
        return (y * np.log(y_pred)) + ((1 - y) * np.log((1 - y_pred)))
    

    def _cost_function(self, y: np.ndarray, y_pred: np.ndarray):
        m = y.shape[0]
        cost = (1/m) * self._loss(y, y_pred)

        return cost
    

    def fit(self, X: np.ndarray, y: np.ndarray):

        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)


            dj_dw = (1/m) * (np.dot(X.T, (y_pred - y)) + ((self._lambda / m) * self.weights))
            dj_db = (1/m) * np.sum(y_pred - y)

            self.weights -= self.weights - self.learning_rate * dj_dw
            self.bias -= self.bias - self.learning_rate * dj_db
    

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)

        return y_pred

    
    def predict(self, X: np.ndarray) -> list[int]:
        y_pred = self.predict_proba(X)

        return [0 if i <= 0.5 else 1 for i in y_pred]