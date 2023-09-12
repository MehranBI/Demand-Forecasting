# =============================================================================
# Import packages
# =============================================================================
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import mlflow
import mlflow.tensorflow

import math
import pyodbc

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import tensorflow as tf


# =============================================================================
# Define parameters
# =============================================================================
use_dummy = 1   # 1 for applying one-hot encoding to categorical features
use_window = 1  # 1 for building windowing features
n_steps = 6    # number of time step to be used in windowing. it works only if use_window = 1
use_scaler = 1  # 1 for applying scaling to x and y
############################################################


# =============================================================================
# Define functions
# =============================================================================
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


#Transform dataframe including timewindows for each freature to 3D np array as an input for LSTM 
#reshape input to be [samples, time steps, features]
def reshape_3D_fun(x,n_steps):
    for i in range( int(x.shape[1] / n_steps) ):
        x_fi = x.iloc[:,i*n_steps:(i+1)*n_steps]
        if i == 0:
            x3D = x_fi.values.reshape( x_fi.shape[0], x_fi.shape[1], 1 )
        else:
            x3D = np.concatenate(
                (
                 x3D,
                 x_fi.values.reshape( x_fi.shape[0], x_fi.shape[1], 1 )
                 ),
                axis = 2
                )         
    return x3D


#create correlation heatmap chart
def corr_heatmap_fun(df):
    plt.figure(figsize=(30,30))
    corr_matrix = df.corr().round(decimals = 2)
    sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)
    plt.show()
    
    return corr_matrix


#train function
def train_fun(tf_model, x_train, y_train, x_validate, y_validate, use_scaler, scaler_x, scaler_y ):
    model_epochs = 30
    model_batch_size = 32
    
    mlflow.log_param("model_epochs", model_epochs)
    mlflow.log_param("model_batch_size", model_batch_size)
    
    tf_model_fit = tf_model.fit(x_train, y_train, epochs = model_epochs, batch_size = model_batch_size, validation_data=(x_validate, y_validate), verbose=2)
    
    plt.figure(figsize=(10,5))
    plt.plot(tf_model_fit.history['loss'], label='train')
    plt.plot(tf_model_fit.history['val_loss'], label='validate')
    plt.legend()
    plt.savefig("learning_curve.png")
    plt.show()
    mlflow.log_artifact("learning_curve.png")
          
    y_train_hat     = tf_model.predict(x_train)
    y_validate_hat  = tf_model.predict(x_validate)
       
    if use_scaler == 1:
        y_train     = scaler_y.inverse_transform( y_train.reshape(-1,1) )
        y_train_hat = scaler_y.inverse_transform( y_train_hat.reshape(-1,1) )
        
        y_validate      = scaler_y.inverse_transform( y_validate.reshape(-1,1) )
        y_validate_hat  = scaler_y.inverse_transform( y_validate_hat.reshape(-1,1) )
    
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
    mlflow.log_metric("r2_train", r2_train)
    print(f"r2 Train: {r2_train:.3%}")
    
    print("---------------")
    
    train_result = pd.DataFrame( np.column_stack( (y_train,y_train_hat) ) ,columns=["Real","Predicted"])
    train_result = train_result.assign(DataSetType = "train")
    
    validate_result = pd.DataFrame( np.column_stack( (y_validate,y_validate_hat) ) ,columns=["Real","Predicted"])
    validate_result = validate_result.assign(DataSetType = "validate")
    
    train_result = pd.concat( [train_result,validate_result], axis = 0).reset_index(drop=True)
    
    return train_result, tf_model_fit
    
#test function
def test_fun(tf_model, x_test, y_test, use_scaler, scaler_x, scaler_y):        
    y_test_hat = tf_model.predict(x_test)
    
    if use_scaler == 1:
        y_test = scaler_y.inverse_transform( y_test.reshape(-1,1) )
        y_test_hat = scaler_y.inverse_transform( y_test_hat.reshape(-1,1) )
    
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
    mlflow.log_metric("r2_test", r2_test)
    print(f"r2 Test: {r2_test:.3%}")
     
    print("---------------")
    
    plt.figure(figsize=(30,15))
    plt.plot(y_test_hat, label='y_test_hat')
    plt.plot(y_test, label='y_test')
    plt.legend()
    plt.savefig("yhat_vs_ytest.png")
    plt.show()
    mlflow.log_artifact("yhat_vs_ytest.png")
    
    # return np.column_stack( (y_test.values,y_test_hat) )
    test_result = pd.DataFrame( np.column_stack( (y_test,y_test_hat) ) ,columns=["Real","Predicted"])
    test_result = test_result.assign(DataSetType = "test")
    return test_result
 

#final report function   
def prediction_report_fun(dim_train,dim_test,train_result,test_result):
    train_report = pd.concat( [dim_train,train_result], axis = 1)
    test_report =  pd.concat( [dim_test,test_result], axis = 1)
    predicted_result = pd.concat([train_report,test_report], axis = 0)
    predicted_result["mape"] = abs( predicted_result["Predicted"] - predicted_result["Real"])/predicted_result["Real"]
    # predicted_result = pd.concat( [x, pd.DataFrame(y_hat, columns=["TotalWeight_hat"])], axis = 1 )
    return predicted_result




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
# =============================================================================
# df = stdev_outlier_fun(df, column_name = "TotalWeight", n_stdev_rolling = 30, coef_stdev=5)

#select not-temporal (ex) and temporal (ts) features
ex_features = ["CalMonthNum","CalWeekNum","CalWeekDayNum","CalMonthDayNum","SpecialDayStatus","DayBeforeStatus","DayAfterStatus","AvgPricePerWeight","CovidLockdownType","WeatherTemp","WeatherWindSpeed","WeatherPrecipitation","WeatherPrecipitationType",]
ts_features = ["TotalWeight","AvgPricePerWeight"]

df_ex = df[ex_features]
df_ts = df[ts_features]

if use_window == 1:
    for df_column_name in df_ts.columns:
        df_ts = window_fun(df_ts, column_name = df_column_name , n_steps = n_steps)
    #remove data before time steps
    df = df[ df.index >= n_steps ].reset_index(drop=True)
    df_ts = df_ts[ df_ts.index >= n_steps ].reset_index(drop=True)
    df_ex = df_ex[ df_ex.index >= n_steps ].reset_index(drop=True)
    
#build dummy features and add them to the main data frame
if use_dummy == 1:
    df_ex = pd.get_dummies(df_ex,
                        columns = df_ex.columns,   
                        prefix_sep='_dum',drop_first = True)


#create train and test dataset:   
df_ex["AvgPricePerWeight"] = df["AvgPricePerWeight"]
x = df_ts.drop(columns = ts_features ,axis = 1)
x_ex = df_ex

#store ex and ts features in an html file
pd.concat( 
            [
            pd.DataFrame(ex_features,columns=["Features"]).assign(FeatureType = "ex"),
            pd.DataFrame(ts_features,columns=["Features"]).assign(FeatureType = "ts")
            ],axis = 0  
        ).reset_index(drop=True).to_html("features.html")

#store ts features after windowing in a html file
x.describe().to_html("lstm_features.html")

y = df["TotalWeight"]  #target variable


split_ratio = 0.80
validate_ratio = 0.90

if use_scaler == 1:
    
    # scaler_x = StandardScaler()
    scaler_x = MinMaxScaler() 
    # scaler_x.fit(x)
    scaler_x.fit( x.iloc[0:math.floor(split_ratio*validate_ratio*x.shape[0]),:] ) #fit scaling based on train dataset
    x = pd.DataFrame( scaler_x.transform(x) , columns = x.columns )
    
    # scaler_x_ex = StandardScaler()
    scaler_x_ex = MinMaxScaler()
    # scaler_x_ex.fit(x_ex)
    scaler_x_ex.fit(x_ex.iloc[0:math.floor(split_ratio*validate_ratio*x_ex.shape[0]),:]) #fit scaling based on train dataset
    x_ex = pd.DataFrame( scaler_x_ex.transform(x_ex) , columns = x_ex.columns )
    
    # scaler_y = StandardScaler() 
    scaler_y = MinMaxScaler()
    #scaler_y.fit( y.to_frame() )
    scaler_y.fit( y[0:math.floor(split_ratio*validate_ratio*y.shape[0])].to_frame() ) #fit scaling based on train dataset
    y = pd.DataFrame( scaler_y.transform(y.to_frame()), columns = y.to_frame().columns ).iloc[:,0]

else:
    scaler_x = 0
    scaler_y = 0


#preparing x for lstm input via reshaping
x3D = reshape_3D_fun(x,n_steps)
x3D = x3D.astype('float32')
x = x3D

x_ex = x_ex.values.astype('float32')

n_features = x.shape[2]

x_train_validate = x[0:math.floor( split_ratio*x.shape[0] ), :,:]
x_train = x_train_validate[0:math.floor( validate_ratio*x_train_validate.shape[0] ), :,:]
x_validate = x_train_validate[math.floor( validate_ratio*x_train_validate.shape[0] ):,:, :]

x_ex_train_validate = x_ex[0:math.floor( split_ratio*x_ex.shape[0] ),:]
x_ex_train = x_ex_train_validate[0:math.floor( validate_ratio*x_ex_train_validate.shape[0] ), :]
x_ex_validate = x_ex_train_validate[math.floor( validate_ratio*x_ex_train_validate.shape[0] ):,:]

x_test  = x[math.floor( split_ratio*x.shape[0] ):,:, :]
x_ex_test  = x_ex[math.floor( split_ratio*x_ex.shape[0] ):, :]

y_train_validate = y.iloc[0:math.floor( split_ratio*y.shape[0] )].reset_index(drop=True)
y_train = y_train_validate.iloc[0:math.floor( validate_ratio*y_train_validate.shape[0] )].reset_index(drop=True)
y_validate = y_train_validate.iloc[math.floor( validate_ratio*y_train_validate.shape[0] ):].reset_index(drop=True)

y_test = y.iloc[math.floor( split_ratio*y.shape[0] ):].reset_index(drop=True)

y_train = y_train.values.astype('float32')
y_validate = y_validate.values.astype('float32')
y_test = y_test.values.astype('float32')


dim = df[ ["PartitionDate","ISCGCompoundCode","TotalWeight"] ]

dim_train = dim.iloc[0:math.floor( split_ratio*dim.shape[0] ),:].reset_index()
dim_test = dim.iloc[math.floor( split_ratio*dim.shape[0] ):,:].reset_index()



# =============================================================================
# Modeling
# =============================================================================
# #Creating model
model_max_features = n_features
model_lstm_unit = 64
model_dense_unit = 64
model_optimizer='adam'
model_loss='mse'
model_activationfun = 'relu'

name_experiment = "TF_DL_experiment"

inp_x = tf.keras.layers.Input( (x.shape[1],x.shape[2]) )
LSTM1 = tf.keras.layers.LSTM(units = 64, return_sequences = True )(inp_x)
LSTM1 = tf.keras.layers.Dropout(0.2)(LSTM1)
LSTM2 = tf.keras.layers.LSTM(units = 64)(LSTM1)
LSTM2 = tf.keras.layers.Dropout(0.2)(LSTM2)

inp_x_ex = tf.keras.layers.Input(x_ex.shape[1])
FC1 = tf.keras.layers.Dense(64,activation = model_activationfun )(inp_x_ex)
FC1 = tf.keras.layers.Dropout(0.2)(FC1)

CONCAT = tf.keras.layers.concatenate( [LSTM2,FC1] )

FC = tf.keras.layers.Dense(128,activation = model_activationfun)(CONCAT)
FC = tf.keras.layers.Dropout(0.2)(FC)

OUT = tf.keras.layers.Dense(1)(FC)

tf_model = tf.keras.Model(
                            inputs = [inp_x,inp_x_ex],
                            outputs = OUT
                            )

tf_model.compile(loss = model_loss, optimizer = model_optimizer )
    

#run the model
mlflow.set_experiment(name_experiment)

mlflow_run_name = 'DL Experiment'

with mlflow.start_run(run_name = mlflow_run_name):


    mlflow.log_artifact("model_features.html")
    mlflow.log_artifact("features.html")
    
    mlflow.log_param("use_dummy", use_dummy)
    mlflow.log_param("use_window", use_window)
    mlflow.log_param("use_scaler", use_scaler)
    mlflow.log_param("n_steps", n_steps)

    mlflow.log_param("n_features", n_features)
    mlflow.log_param("model_lstm_unit", model_lstm_unit)
    mlflow.log_param("model_dense_unit", model_dense_unit)
    mlflow.log_param("model_optimizer", model_optimizer)
    mlflow.log_param("model_loss", model_loss)
    
    train_result, train_history = train_fun(tf_model, [x_train,x_ex_train], y_train, [x_validate,x_ex_validate], y_validate, use_scaler, scaler_x, scaler_y )
    
    test_result = test_fun(tf_model, [x_test,x_ex_test], y_test, use_scaler, scaler_x, scaler_y )
    
    prediction_report = prediction_report_fun(dim_train,dim_test,train_result,test_result)
        
    mlflow.tensorflow.log_model(tf_model, "log_DL_model")
    
    # tf_model.summary()
    
    #drawing model network and logging it
    tf.keras.utils.plot_model(tf_model, "LSTM_Ex_model_Network.png",show_shapes=True)
    mlflow.log_artifact("LSTM_Ex_model_Network.png")
       
    print("Model run: ", mlflow.active_run().info.run_uuid)
    
mlflow.end_run()

prediction_report.to_excel("prediction_report.xlsx")
