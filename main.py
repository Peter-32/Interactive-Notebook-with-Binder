import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import StandardScaler
from pandasql import sqldf
pd.set_option('display.mpl_style', 'default')
pd.set_option('display.line_width', 5000)
pd.set_option('display.max_columns', 60)
%matplotlib inline
q = lambda q: sqldf(q, globals())
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, neighbors, tree, svm, ensemble
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import datasets
mpl.rcParams['figure.figsize'] = (15.0, 5.0)
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['y'] = iris.target
df['y'] = df['y'].apply(lambda x: str(x))
df


# Heatmap
cor_mat = df.corr().apply(lambda x: abs(x))
sns.heatmap(cor_mat, vmax=.8, square=True)
# Pairplot
sns.pairplot(df, hue="y", palette="Set2", diag_kind="kde", size=2.5)




from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(df, 'y', colormap=plt.get_cmap("Set2"))



ax = sns.tsplot(data=df)
