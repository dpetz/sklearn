
import os
while (not os.getcwd().endswith('clv-predictor')):
    os.chdir('..') # Set Python's current working folder to project folder when run from sub folder 

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from importlib import reload
from markets import db, ds

exa = "EXA_DI"