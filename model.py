"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = feature_vector_df[[['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h',
       'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
       'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
       'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h',
       'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
       'Barcelona_rain_3h']]]
    df=df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year   
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day       
    df['hour'] = df['time'].dt.hour

    df_test1['time'] = pd.to_datetime(df_test1['time'])
    df_test1['year'] = df_test1['time'].dt.year   
    df_test1['month'] = df_test1['time'].dt.month
    df_test1['day'] = df_test1['time'].dt.day       
    df_test1['hour'] = df_test1['time'].dt.hour

    df['Valencia_pressure'] = df['Valencia_pressure'].fillna(df['Valencia_pressure'].mode()[0])
    df_test1['Valencia_pressure'] = df_test1['Valencia_pressure'].fillna(df_test['Valencia_pressure'].mode()[0])
    
    df['Valencia_wind_deg'] = df['Valencia_wind_deg'].str.extract('(\d+)')
    df['Seville_pressure'] = df['Seville_pressure'].str.extract('(\d+)')

    df['Valencia_wind_deg'] = pd.to_numeric(df['Valencia_wind_deg'])
    df['Seville_pressure'] = pd.to_numeric(df['Seville_pressure'])

    df_test1['Seville_pressure'] = df_test1['Seville_pressure'].str.extract('(\d+)')
    df_test1['Valencia_wind_deg'] = df_test1['Valencia_wind_deg'].str.extract('(\d+)')
    
    df_test1['Seville_pressure'] = pd.to_numeric(df_test1['Seville_pressure'])
    df_test1['Valencia_wind_deg'] = pd.to_numeric(df_test1['Valencia_wind_deg'])

    from sklearn.preprocessing import StandardScaler

    #standardize the train and test data
    standardized_train = df.drop(['load_shortfall_3h','time'], axis=1)
    standardized_test=df_test1.drop('time',axis=1)

    # create scaler object
    scaler = StandardScaler()

    test_scaled = scaler.fit_transform(standardized_test)
    train_scaled = scaler.fit_transform(standardized_train)

    # convert the scaled predictor values into a dataframe
    stand_train = pd.DataFrame(train_scaled,columns=standardized_train.columns)
    stand_test = pd.DataFrame(test_scaled,columns=standardized_test.columns)
   
    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    X = df.drop(['load_shortfall_3h','time'], axis=1)
    y = df['load_shortfall_3h']
    X_train, X_test, y_train, y_test = train_test_split(standardized_train,y,test_size=0.20,random_state=1)
    prep_data = _preprocess_data(data)


    # Perform prediction with model and preprocessed data.
    regr_tree = DecisionTreeRegressor(max_depth=4,random_state=42)
    regr_tree.fit(x_train,y_train)
    # Test the model
    y_pred_train = regr_tree.predict(x_train)
    y_pred_test = regr_tree.predict(x_test)
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
