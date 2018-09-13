import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier 
import pandas as pd
from markets import ds


class TreeImputer(BaseEstimator, TransformerMixin):
    """ Replaces all missing values by decision tree predictions.

    When calling ``fit`` all variables with at least on missing variables
    are idenitfied. For each variable a decision tree is trained on all
    rows with available values and using all numeric variables without missing
    values as inputs.

    If the variable to impute is numeric a ``DecisionTreeRegressor`` is trained,
    otherwise a ``DecisionTreeClassifier``.

    Any parameters passed to the constructor of this class are forwarded
    as parameters to the underlying decision trees.

    TODO: Implement w/o pandas, just numpy

    See also
    --------
    http://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator

    """

    def __init__(self,columns=None, **params):
        self.columns = columns
        self.spec = {'criterion':'mse','max_leaf_nodes':16,'max_features':'log2'}
        self.spec.update(params)
        self.regression_tree = DecisionTreeRegressor(**self.spec)
        self.classification_tree = DecisionTreeClassifier(**self.spec)
    
    def fit(self, X, y=None):
     
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X.toarray())

        missings = (X.shape[0] - X.describe().loc['count'].transpose()).astype(int)

        self.miss_vars = missings[missings > 0].index
        if self.columns is not None:
            self.miss_vars = set(self.miss_vars).intersection(set(self.columns))
        self.full_vars = (X[missings[missings == 0].index].dtypes == 'float64').index   
        def train_tree(miss_var):
            notna = pd.notna(X[miss_var])
            no_outl = ds.filter_std(X,miss_var)
            print("Training tree for %s (%i inputs, %i rows)..." % (miss_var,len(self.full_vars),notna.shape[0]) )
            spec = self.regression_tree if (X[miss_var].dtype == 'float64') else self.classification_tree
            return spec.fit(no_outl[self.full_vars],no_outl[miss_var])
        
        self.models = dict([(v,train_tree(v)) for v in self.miss_vars])

        return self

    def transform(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['models'])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X.toarray())

        #print("... Imputing: %s" % (self.miss_vars))
    
        for var in self.miss_vars:
            X_ = X.loc[np.isnan(X[var]),self.full_vars]
            if (X_.shape[0] > 0):
                X.loc[np.isnan(X[var]),var] = self.models[var].predict(X_)    
        return X