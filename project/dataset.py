import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from config import config, initializing
initializing()


# Attention module
#------------------------------------------------------------------------------------------------------

def calculate_mut_info(input_data, sensitive_feature, feature_class):
    """
    Summary:
        calculate Mutual Information 
    Arguments:
        input_data (array): dataset
        sensitive_feature (string): name of the sensitive feature
        feature_class (string): name of the class
    Return:
        Mutual information
    """
    
    data = input_data[(input_data[sensitive_feature]==feature_class)]
    sf_list = ['race', 'sex']
    sensitive_data = data.loc[:, sf_list]
    sf_list.remove(sensitive_feature)
    sensitive_data = sensitive_data.drop(columns=sf_list)
    data = data.drop(columns=['target', 'race', 'sex'])
    mut_data = mutual_info_classif(data, sensitive_data[sensitive_feature])
    return mut_data

def attentionModule(input_data, threshold):
    """
    Summary:
        Attention Module select features based on Mutual Information
    Arguments:
        input_data (pandas.DataFrame): dataaset
        threshold (float): Threshold Aggregator
    Return:
        list of remove features name
    """
    
    input_data = input_data.dropna()
    # encoding the features that has class values
    lb_make = LabelEncoder()
    input_data['workclass'] = lb_make.fit_transform(input_data['workclass'])
    input_data['education'] = lb_make.fit_transform(input_data['workclass'])
    input_data['marital_status'] = lb_make.fit_transform(input_data['marital_status'])
    input_data['occupation'] = lb_make.fit_transform(input_data['occupation'])
    input_data['relationship'] = lb_make.fit_transform(input_data['relationship'])
    input_data['country'] = lb_make.fit_transform(input_data['country'])
    input_data['target'] = lb_make.fit_transform(input_data['target'])

    # mutual information calculate
    mut_race_white = calculate_mut_info(input_data, 'race', 'White')
    mut_race_black = calculate_mut_info(input_data, 'race', 'Black')
    mut_sex_male = calculate_mut_info(input_data, 'sex', 'Male')
    mut_sex_female = calculate_mut_info(input_data, 'sex', 'Female')

    # Feature selection based on threshold
    mut_race = mut_race_white * mut_race_black
    mut_sex = mut_sex_male * mut_sex_female
    mut = mut_race + mut_sex
    mut_percent = (mut/np.sum(mut))*100
    remove_features = []
    features_name = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
                'marital_status', 'occupation', 'relationship','capital_gain',
                'capital_loss', 'hours_per_week', 'country']

    for i in range(len(features_name)):
        if mut_percent[i] < threshold:
            remove_features.append(features_name[i])

    return remove_features


# load data
#------------------------------------------------------------------------------------------------------

def load_ICU_data(path, threshold=0):
    """
    Summary:
        read and pre-process the data
    Arguments:
        path (string): dataset path
        threshold (float): Threshold Aggregator. By default its value is 0 mean no threshold value is provided than all the feature will be taken
    Return:
        pre-processed dataset
    """
    
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
                    'marital_status', 'occupation', 'relationship', 'race', 'sex', 
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names, 
                              na_values="?", sep=r'\s*,\s*', engine='python') # here seperator -- 0 or more whitespace then , then 0 or more whitespace --
                              .loc[lambda df: df['race'].isin(['White', 'Black'])])

    remove_features = attentionModule(input_data, threshold)
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['race', 'sex']
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(race=lambda df: (df['race'] == 'White').astype(int),
                 sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'] == '>50K').astype(int) # Cast a pandas object to a specified dtype dtype.

    # features; note that the 'target' and sentive attribute columns are dropped
    remove_features.extend(['target', 'race', 'sex'])
    X = (input_data
         .drop(columns=remove_features)
         .fillna('Unknown')   # The fillna() function is used to fill NA/NaN values using the specified method
         .pipe(pd.get_dummies, drop_first=True)) # Use .pipe when chaining together functions that expect Series, DataFrames or GroupBy objects.
                                                 # pd.get_dummies=Convert categorical variable into dummy/indicator variables
                                                 # drop_first: bool function(default False) Whether to get k-1 dummies out of k categorical levels by removing the first level.
    
    
    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape[0]} samples")
    print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
    return X,  y, Z
