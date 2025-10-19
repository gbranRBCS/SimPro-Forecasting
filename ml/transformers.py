from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class RareCategoryCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns, top_k=20, other_label="OTHER"):
        self.columns = columns
        self.top_k = top_k
        self.other_label = other_label
        self.keep_maps = {}

    def fit(self, X, y=None):
        # X is a DataFrame
        for c in self.columns:
            vc = X[c].value_counts(dropna=False)
            self.keep_maps[c] = set(vc.head(self.top_k).index.tolist())
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            keep = self.keep_maps.get(c, set())
            X[c] = X[c].where(X[c].isin(keep), other=self.other_label)
        return X
