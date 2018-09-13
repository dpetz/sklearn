# https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696
# Pandas integration: https://github.com/scikit-learn/scikit-learn/issues/5523
# More an panda based transformers: https://www.kaggle.com/jankoch/scikit-learn-pipelines-and-pandas 

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class Fork(BaseEstimator, TransformerMixin):
    def __init__(self, col_partitions, *pipelines):
        self.col_partitions = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.select_dtypes(include=[self.dtype])
        