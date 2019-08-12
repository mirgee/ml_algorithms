from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS:

    def __init__(self, estimator, k, scoring=accuracy_score,
                 validation_size=0.25, random_seed=1):
        self.estimator = estimator
        self.k = k
        self.scoring = scoring
        self.validation_size = validation_size
        self.random_seed = 1

    # TODO: Record scores, chosen subsets
    def fit(self, X, y):
        self.final_score_ = None
        self.scores_ = []
        self.subsets_ = []

        X_train, X_test, y_train, y_test = \
            train_test_split(
                X, y, test_size=self.validation_size,
                random_state=self.random_seed)

        d = X_train.shape[1]
        self.feature_indeces_ = tuple(range(d))

        while d > self.k:
            scores = []
            subsets = []

            for sub in combinations(self.feature_indeces_, d-1):
                score = \
                        self._calculate_score(sub, X_train, X_test, y_train,
                                              y_test)
                scores.append(score)
                subsets.append(sub)

            highest_scoring = np.argmax(scores)
            self.feature_indeces_ = subsets[highest_scoring]
            self.scores_.append(scores[highest_scoring])
            self.subsets_.append(self.feature_indeces_)

            d -= 1

        self.final_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.feature_indeces_]

    def _calculate_score(self, sub, X_train, X_test, y_train, y_test):
        self.estimator.fit(X_train[:, sub], y_train)
        y_pred = self.estimator.predict(X_test[:, sub])
        return self.scoring(y_test, y_pred)
