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

    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
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

    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape[0]} samples")
    print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
    return X, y, Z


def scale_df(df, scaler):
    return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)


def data_split(X, y, z):
    # split into train/test set
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

    X, y, z = read_data(config["dataset_dir"])

    class_weights = compute_class_weights(z)

    X_train, X_val, X_test, y_train, y_val, y_test, Z_val, Z_train, Z_test = data_split(
        X, y, z)

    # standardize the data
    scaler = StandardScaler().fit(X_train)
    X_train = X_train.pipe(scale_df, scaler)
    X_val = X_val.pipe(scale_df, scaler)
    X_test = X_test.pipe(scale_df, scaler)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).batch(config["batch_size"])
    Z_train_ds = tf.data.Dataset.from_tensor_slices(
        (Z_train)).batch(config["batch_size"])

    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)).batch(config["batch_size"])
    Z_val_ds = tf.data.Dataset.from_tensor_slices(
        (Z_val)).batch(config["batch_size"])

    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)).batch(config["batch_size"])
    Z_test_ds = tf.data.Dataset.from_tensor_slices(
        (Z_test)).batch(config["batch_size"])

    if test == True:
        return test_ds, Z_test_ds

    return train_ds, Z_train_ds, val_ds, Z_val_ds, class_weights
    # for x, y in train_ds:
    #     print(x[0], y[1])
    #     break


# get_dataloader()
