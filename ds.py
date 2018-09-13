# Data Science utility functions

from sklearn.externals import joblib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score,confusion_matrix
import matplotlib.pyplot as plt

def split(df,cols,drop=None):
    """ Drop columns from dataframe and returns both as separate dataframes.
        Optionally drop some columns from both.
        Typical use case: ``X,y = ds.split(df,'target_var',drop='id_var')``
    """
    cols = [cols] if isinstance(cols, str) else cols
    df = (df.drop(drop,axis=1) if drop else df)
    return (df.drop(cols,axis=1),df[cols])

def select_number(X):
    return X.select_dtypes(include='number')

def select_object(X):
    return X.select_dtypes(include='object')

def var_missings(df):
    return ((len(df) - df.describe().loc['count'].transpose()).astype(int).sort_values(ascending=False))

def filter_std(df,col,dist=3):
    """Filters all rows from data frame which values of ``col`` are more than ``dist``
       standard devisations away from the mean """ 
    return df[np.abs(df[col]-df[col].mean())<=(dist*df[col].std())]

def load_model(model_id):
    """Loads pipeline file by id"""
    return joblib.load("model/%s/Model.pkl" % (model_id))

def save_model(model,model_id):
    """Saves pipeline file by given id"""
    joblib.dump(model, "model/%s/Model.pkl" % (model_id)) 

def roc(y_true,y_proba):
    """Draws ROC chart""" 
    fpr, tpr, thresholds = roc_curve(y_true, y_proba,pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
         label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Curve')
    plt.legend(loc="lower right")
    plt.show()

def metrics(y_true,y_score,metrics=[roc_auc_score,confusion_matrix]):
    """Calculates set of metrics and returns as dict """
    return dict([(m.__name__,m(y_true,y_score)) for m in metrics])

def load_scores(model_id,data_id):
    scores_dir = "model/%s/%s/" % (model_id, data_id)

    ids = scores_dir + 'Ids.desc'
    if not os.path.exists(ids):
        ids = scores_dir + 'Ids'

    scores = pd.read_csv(ids,header=None)
    scores[0] = scores[0].astype(int)
    proba = pd.read_csv(scores_dir +'True.proba',header=None,skipinitialspace=True)
    scores[model_id] = proba.astype('float64')

    scores.set_index(0,inplace=True)
    scores.index.rename('Id', inplace=True) 
    return scores   

class ScoreExport:
    """Persists model results for given model and data ids."""

    def __init__(self,scores_df,model_id,data_id,ranked=False):
        self.scores_df = scores_df
        self.export_dir = ("model/%s/%s/") % (model_id, data_id)
        self.id_file = 'ids' + ('.desc' if ranked else '')

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
 
    def __export_columns(self,file_name,columns,idx=False,fmt='%10.5f'):
        self.scores_df.to_csv(self.export_dir+file_name,index=idx,
            header=False,float_format=fmt,columns=columns)

    def __enter__(self):
        """Initialize export by eporting row ids"""
        self.__export_columns(self.id_file,[],idx=True,fmt='%10.0f')
        return self

    def predictions(self,column):
        """ Exports predictions from given column to file"""
        self.__export_columns('predicions',[column])

    def probabilities(self,column):
        """ Exports class posterior probabilities from given column to file"""    
        self.__export_columns("%s.proba" % column,[column])

    def __exit__(self, type, value, traceback):
        """Final checks and success message"""
        print("...\n%i customers scores exported to '%s'." % (len(self.scores_df),self.export_dir))
