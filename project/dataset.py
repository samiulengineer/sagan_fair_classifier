import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import config, initializing
initializing()


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
