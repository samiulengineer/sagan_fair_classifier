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
    data = input_data[(input_data[sensitive_feature]==feature_class)]
    sf_list = ['race', 'sex']
    sensitive_data = data.loc[:, sf_list]
    sf_list.remove(sensitive_feature)
    sensitive_data = sensitive_data.drop(columns=sf_list)
    data = data.drop(columns=['target', 'race', 'sex'])
    mut_data = mutual_info_classif(data, sensitive_data[sensitive_feature])
    return mut_data

def attentionModule(input_data, threshold):
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
    X_proxy = (input_data
         .drop(columns=remove_features)
         .fillna('Unknown')   # The fillna() function is used to fill NA/NaN values using the specified method
         .pipe(pd.get_dummies, drop_first=True)) # Use .pipe when chaining together functions that expect Series, DataFrames or GroupBy objects.
                                                 # pd.get_dummies=Convert categorical variable into dummy/indicator variables
                                                 # drop_first: bool function(default False) Whether to get k-1 dummies out of k categorical levels by removing the first level.
    
    
    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape[0]} samples")
    print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
    return X, y, Z



def read_data(path):

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python')
                  .loc[lambda df: df['race'].isin(['White', 'Black'])])

    return input_data


def cleaning_data(input_data):

    input_data = input_data.dropna()  # remove null values

    input_data = input_data.drop_duplicates()  # remove all duplicate value

    return input_data


def get_sensitive_attrib(input_data):

    sensitive_attribs = ['race', 'sex']
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(race=lambda df: (df['race'] == 'White').astype(int),
                 sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'] == '>50K').astype(int)  # .to_frame()

    # features; note that the 'target' and sentive attribute columns are dropped
    X = (input_data
         .drop(columns=['target', 'race', 'sex'])
         .fillna('Unknown').pipe(pd.get_dummies, drop_first=True))

    return X, y, Z


def scale_df(df, scaler):

    return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)


def data_split(X, y, z):

    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, z, test_size=config["split_size"],
                                                                         stratify=y, random_state=7)

    X_val, X_test, y_val, y_test, Z_val, Z_test = train_test_split(X_test, y_test, Z_test, test_size=0.5,
                                                                   stratify=y_test, random_state=7)

    return X_train, X_val, X_test, y_train, y_val, y_test, Z_val, Z_train, Z_test


def compute_class_weights(data_set, classes=[0, 1]):

    class_weights = []
    if len(data_set.shape) == 1:
        balanced_weights = compute_class_weight(
            'balanced', classes=classes, y=data_set)
        class_weights.append(dict(zip(classes, balanced_weights)))
    else:
        n_attr = data_set.shape[1]
        for attr_idx in range(n_attr):
            balanced_weights = compute_class_weight('balanced', classes=classes,
                                                    y=np.array(data_set)[:, attr_idx])
            class_weights.append(dict(zip(classes, balanced_weights)))

    # converting into single dict
    cls_weight = {0: 1, 1: 1}
    for i in class_weights:
        cls_weight[0] = i[0]*cls_weight[0]
        cls_weight[1] = i[1]*cls_weight[1]

    return cls_weight


def get_dataloader(test=False):

    # read dataset
    input_data = read_data(config["dataset_dir"])

    # cleaning the dataset
    input_data = cleaning_data(input_data)

    X, y, z = get_sensitive_attrib(input_data)
    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape[0]} samples")
    print(f"sensitives Z: {z.shape[0]} samples, {z.shape[1]} attributes")

    class_weights = compute_class_weights(z)

    X_train, X_val, X_test, y_train, y_val, y_test, Z_val, Z_train, Z_test = data_split(
        X, y, z)

    # standardize the data
    scaler = StandardScaler().fit(X_train)
    X_train = X_train.pipe(scale_df, scaler)
    X_val = X_val.pipe(scale_df, scaler)
    X_test = X_test.pipe(scale_df, scaler)

    if test == True:
        return X_test, y_test, Z_test

    return X_train, y_train, X_val, y_val, Z_train, Z_val          # , class_weights


# get_dataloader()
