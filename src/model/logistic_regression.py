import numpy as np

class LogisticRegressionScratch():
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.iterations = iterations


    def _sigmoid(self, z: float):
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid
    
    def _loss(self, y: int, y_pred: int):
        return (y * np.log(y_pred)) + ((1 - y) * np.log((1 - y_pred)))
    
    def _cost_function(self, y: np.ndarray, y_pred: np.ndarray):
        m = y.shape[0]
        cost = 0
        for i in range(m):
            cost += self._loss(y[i], y_pred[i])
        cost = (1/m) * cost
        return cost
    
    def _gradient(self, X: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in: float) -> list[np.ndarray | float]:
        
        m = X.shape[0]
        dj_dw = np.zeros(w_in)
        dj_db = 0


        
        return dj_dw, dj_db

    def fit(self, X: np.ndarray, y: np.ndarray):

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for i in range(self.iterations):
            z = np.dot()

        return None
    
    def predict(self, X):
        return None