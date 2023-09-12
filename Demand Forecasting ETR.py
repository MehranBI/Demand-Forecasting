#Importing Basic Packages:
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#importing mlflolw:
import mlflow
import mlflow.sklearn

#Importing Additional Packages:
import math
import pyodbc

#Importing sklearn Maodules:
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

############################################################


#####################Input parameters########################
use_dummy = 0   # 1 for applying one-hot encoding to categorical features
use_window = 1  # 1 for building windowing features
n_steps = 6     # number of time step to be used in windowing. it works only if use_window = 1
use_scaler = 0  # 1 for applying scaling to x and y
############################################################

#############################MFunctions##################################

#detect and remove outliers
def stdev_outlier_fun(df, column_name, n_stdev_rolling, coef_stdev):
    # df[f'{column_name}_stdev_{n_stdev_rolling}'] = df[column_name].rolling(n_stdev_rolling).std()
    # df[f'{column_name}_avg_{n_stdev_rolling}'] = df[column_name].rolling(n_stdev_rolling).mean()
    df["up"]  = df[column_name].rolling(n_stdev_rolling).mean() + coef_stdev* df[column_name].rolling(n_stdev_rolling).std() - df[column_name]
    df["low"] = df[column_name]-( df[column_name].rolling(n_stdev_rolling).mean() - coef_stdev* df[column_name].rolling(n_stdev_rolling).std() )    
    return df[ (df["low"] >0) & (df["up"] >0) ].drop( columns = ["up","low"] ).reset_index(drop=True) 


#create time steps features and adding them to the main data frame
def window_fun(df, column_name, n_steps):
    for i in range(1,n_steps+1):
        df[f'{column_name}_{i}'] = df[column_name].shift(i)
    return df


#create correlation heatmap chart
def corr_heatmap_fun(df):
    plt.figure(figsize=(30,30))
    corr_matrix = df.corr().round(decimals = 2)
    sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)
    plt.show()
    
    return corr_matrix

#function for scaling data
def MinMaxScalerTransform_fun(x,df_min,df_max):
    x_scaled = (x - df_min) / (df_max - df_min)
    
    return x_scaled

#function for inverse scaling data
def MinMaxScalerInverse_fun(x_scaled,df_min,df_max):
    x = x_scaled*(df_max - df_min) + df_min   
    return x

#train function
def train_fun(sk_model, x_train, y_train):
    sk_model = sk_model.fit(x_train, y_train)  
    y_train_hat = sk_model.predict(x_train)
    
    mape_train = mean_absolute_percentage_error(y_train,y_train_hat)
    mlflow.log_metric("mape_train", mape_train)
    print(f"mape train: {mape_train:.3%}")
    
    mae_train = mean_absolute_error(y_train,y_train_hat)
    mlflow.log_metric("mae_train", mae_train)
    print(f"mae train: {mae_train}")  
    
    rmse_train = math.sqrt(mean_squared_error(y_train,y_train_hat))
    mlflow.log_metric("rmse_train", rmse_train)
    print(f"rmse train: {rmse_train}")  
    
    r2_train = r2_score(y_train,y_train_hat)
    # r2_train = sk_model.score(x_train, y_train)
    mlflow.log_metric("r2_train", r2_train)
    print(f"r2 train: {r2_train:.3%}")
    
    print("---------------")
    
    train_result = pd.DataFrame( np.column_stack( (y_train.values,y_train_hat) ) ,columns=["Real","Predicted"])
    train_result = train_result.assign(DataSetType = "train")
    return train_result
    
#test function
def test_fun(sk_model, x_test, y_test):
    y_test_hat = sk_model.predict(x_test)
    
    mape_test = mean_absolute_percentage_error(y_test,y_test_hat)
    mlflow.log_metric("mape_test", mape_test)
    print(f"mape test: {mape_test:.3%}")
    
    mae_test = mean_absolute_error(y_test,y_test_hat)
    mlflow.log_metric("mae_test", mae_test)
    print(f"mae test: {mae_test}")
    
    rmse_test = math.sqrt(mean_squared_error(y_test,y_test_hat))
    mlflow.log_metric("rmse_test", rmse_test)
    print(f"rmse test: {rmse_test}")
    
    r2_test = r2_score(y_test,y_test_hat)
    # r2_test = sk_model.score(x_test, y_test)
    mlflow.log_metric("r2_test", r2_test)
    print(f"r2 test: {r2_test:.3%}")

    
    print("---------------")
    
    plt.figure(figsize=(30,15))
    plt.plot(y_test_hat, label='y_test_hat')
    plt.plot(y_test.values, label='y_test')
    plt.legend()
    plt.savefig("yhat_vs_ytest.png")
    plt.show()
    mlflow.log_artifact("yhat_vs_ytest.png")
    
    test_result = pd.DataFrame( np.column_stack( (y_test.values,y_test_hat) ) ,columns=["Real","Predicted"])
    test_result = test_result.assign(DataSetType = "test")
    return test_result
 

#feature importance function   
def feature_imp_fun(sk_model):
    feature_names = sk_model.feature_names_in_
    feature_imp = sk_model.feature_importances_
    return pd.DataFrame( np.column_stack( (feature_names,feature_imp) ), columns=["Feature","Importance"])

#final report function   
def prediction_report_fun(dim_train,dim_test,train_result,test_result):
    train_report = pd.concat( [dim_train,train_result], axis = 1)
    test_report =  pd.concat( [dim_test,test_result], axis = 1)
    predicted_result = pd.concat([train_report,test_report], axis = 0)
    predicted_result["mape"] = abs( predicted_result["Predicted"] - predicted_result["Real"])/predicted_result["Real"]
    return predicted_result

###############################################################


# =============================================================================
# Read data
# =============================================================================
cnxn = pyodbc.connect('Driver={?};'
                      'Server=?;'
                      'Database=?;'
                      'Trusted_Connection=yes;')


Query = 'select * from ?'

df = pd.read_sql_query( Query ,cnxn)
df_head = df.head()
df_describe = df.describe()
df_shape = df.shape
df_type = df.dtypes

df_min = df.min(numeric_only=True).min()
df_max = df.max(numeric_only=True).max()


# =============================================================================
# Prepare Data
# =============================================================================#deleting outliers
       
if use_window == 1:
    df = window_fun(df, column_name = "TotalWeight", n_steps = n_steps)
    df = window_fun(df, column_name = "AvgPricePerWeight", n_steps = n_steps)
    df = df[ df.index >= n_steps ].reset_index(drop=True)
    

if use_dummy == 1:
    df = pd.get_dummies(df, 
                        columns=["CalMonthNum","CalWeekNum","CalWeekDayNum","CalMonthDayNum","SpecialDayStatus","DayBeforeStatus","DayAfterStatus","CovidLockdownType","WeatherPrecipitationType"],
                        prefix_sep='_dum',drop_first = False)

#create train and test dataset:   
x = df.drop(columns = ["PartitionDate", "TotalWeight"], axis=1)  #features

moldel_features = x.describe().to_html("model_features.html")

y = df["TotalWeight"]  #target variable

if use_scaler == 1:
    scaler_x = MinMaxScaler()
    scaler_x.fit(x)
    x = pd.DataFrame( scaler_x.transform(x) , columns = x.columns )
    
    scaler_y = MinMaxScaler()
    scaler_y.fit(y.to_frame())
    y = pd.DataFrame( scaler_y.transform(y.to_frame()), columns = y.to_frame().columns ).iloc[:,0]


dim = df[ ["PartitionDate","TotalWeight"] ]

n_features = x.shape[1]
split_ratio = 0.80

x_train = x.iloc[0:math.floor( split_ratio*df.shape[0] ), :].reset_index(drop=True)
x_test = x.iloc[math.floor( split_ratio*df.shape[0] ):, :].reset_index(drop=True)

y_train = y.iloc[0:math.floor( split_ratio*df.shape[0] )].reset_index(drop=True)
y_test = y.iloc[math.floor( split_ratio*df.shape[0] ):].reset_index(drop=True)

dim_train = dim.iloc[0:math.floor( split_ratio*df.shape[0] ),:].reset_index()
dim_test = dim.iloc[math.floor( split_ratio*df.shape[0] ):,:].reset_index()


# =============================================================================
# Modeling
# =============================================================================#Creating model
name_experiment = "scikit_ETR_experiment"

"""
##Parameter tunning
max_depth = list(range(10,100)) + [None]
min_samples_split = range(5,20)
min_samples_leaf = range(2,10)
max_features = [0.5, 0.6, 0.7, 0.8, 0.9] #range(5,13)
bootstrap = [False,True]
max_samples = [0.8,0.85,0.9,0.95]

param_dist = {'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'max_features': max_features,
'bootstrap': bootstrap,
'max_samples': max_samples}

tscv = TimeSeriesSplit(n_splits=3)
ETR_cv = RandomizedSearchCV(sk_model, param_dist, cv=tscv, verbose=1, n_jobs=-1, n_iter=400, scoring='neg_mean_squared_error')
ETR_cv.fit(x_train,y_train)
print('Tuned Forest Parameters:', ETR_cv.best_params_)
r2_score(y_train,ETR_cv.predict(x_train))
"""

sk_model = ExtraTreesRegressor(n_jobs=-1, 
                                n_estimators=300, 
                                min_samples_split=10, 
                                min_samples_leaf=2, 
                                max_samples=0.95, 
                                max_features=0.5, 
                                max_depth=100, 
                                bootstrap=True)

#running the model
mlflow.set_experiment(name_experiment)

mlflow_run_name = 'ETR_Experiment'
    
with mlflow.start_run(run_name = mlflow_run_name):

    mlflow.log_artifact("model_features.html")    
    mlflow.log_param("use_dummy", use_dummy)
    mlflow.log_param("use_window", use_window)
    mlflow.log_param("use_scaler", use_scaler)
    mlflow.log_param("n_steps", n_steps)

    mlflow.log_param("n_features", n_features)
    
    train_result = train_fun(sk_model, x_train, y_train)
    test_result = test_fun(sk_model, x_test, y_test)
    
    feature_imp = feature_imp_fun(sk_model)
    feature_imp.to_html("feature_imp.html")
    mlflow.log_artifact("feature_imp.html")
    
    
    prediction_report = prediction_report_fun(dim_train,dim_test,train_result,test_result)

    mlflow.sklearn.log_model(sk_model, "log_model")
    print("Model run: ", mlflow.active_run().info.run_uuid)
    
mlflow.end_run()

prediction_report.to_excel("prediction_report.xlsx")