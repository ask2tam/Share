
# !conda install r-essentials
# !conda install rpy2 tzlocal
# Otra forma es instalar compilados .whl de http://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2 con !pip install rpy2-2.9.4-cp36-cp36m-win_amd64.whl
# Luego instalar las librerias necesarias en R.exe 
# !pip install msgpack
# !pip install shap xgboost eli5 treeinterpreter lime itertools graphviz pydotplus scikit-optimize pycebox pdpbox

# Librerias basicas
import shap
import xgboost
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import eli5
import warnings
import treeinterpreter.treeinterpreter as ti
import lime
import itertools
import graphviz
import pydotplus
import gc

from sklearn.preprocessing import Imputer
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from skll.metrics import spearman
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from eli5.sklearn import PermutationImportance
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from IPython.display import Image  
from io import StringIO  
from concurrent.futures import ProcessPoolExecutor
from matplotlib.colorbar import ColorbarBase
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pycebox.ice import ice, ice_plot
from pdpbox import pdp
from lime.lime_tabular import LimeTabularExplainer

# Librerias de rpy2
import os
os.environ['R_HOME'] = 'C:/Users/s11250/AppData/Local/Continuum/anaconda3/Lib/R'
os.environ['R_USER'] = 'C:/Users/s11250/AppData/Local/Continuum/anaconda3/Lib/R/library'
from rpy2 import robjects
from rpy2.robjects import Formula, Environment, r, pandas2ri
from rpy2.robjects.vectors import IntVector, FloatVector
from rpy2.robjects.lib import grid
from rpy2.robjects.packages import importr, data
from rpy2.rinterface import RRuntimeError
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


# Ejemplo1
m = r.matrix(r.rnorm(100), ncol=5)
pca = r.princomp(m)
# Ejemplo2
ctl = r.c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
trt = r.c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
group = r.gl(2, 10, 20, labels = ["Ctl","Trt"])
weight = ctl + trt
robjects.globalenv["weight"] = weight
robjects.globalenv["group"] = group
lm_D9 = r.lm("weight ~ group")
print(r.anova(lm_D9))
# Ejemplo3
r(''' 
f <- function(r, verbose=FALSE) {r * pi}
f(3)
''')
# Ejemplo4
xgboost = importr('xgboost') # Cargar la libreria R
os.chdir('D:/_mtam/xgboost/SimRun')
print(os.getcwd())

xgbModel = r.readRDS("DISEF_mvp3_best.rds")
explicative_vars = xgbModel.rx('variables')[0]
xgbModel = xgbModel.rx('model')[0]

data=pd.read_csv('development_num_base.csv')
dt_sim=data[pandas2ri.ri2py(explicative_vars)].as_matrix()
preds = r.predict(xgbModel, dt_sim)
preds = pandas2ri.ri2py(preds)

# Refresh RAM
r.gc() # R
gc.collect() # Python



