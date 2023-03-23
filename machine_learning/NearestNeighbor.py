import numpy as np
from scipy.stats import mode
from sklearn.metrics import f1_score

class KNN:
    def __init__(self,n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def eucledian_distance(self, given_input):
        return np.sqrt(np.sum((self.X_train - given_input)**2,axis=1))

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = self.eucledian_distance(x)
            index_of_nearest_k = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            y_pred.append(mode(self.y_train[index_of_nearest_k])[0][0])
        return y_pred

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return f1_score(y_pred, y_test)