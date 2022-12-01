
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from config import config, initializing
initializing()


# P% rule metrics
#------------------------------------------------------------------------------------------------------

def p_rule(y_pred, z_values, threshold=0.5):
    """
    Summary:
        calculate p%-rule
    Arguments:
        y_pred (pandas.DataFrame): predicted  target
        z_values (pandas.DataFrame): sensitive features
        threshold (float): threshold for dividing the class
    Return:
        p%-rule value
    """
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1/odds]) * 100


# Fairness metrics for DI, EOp, EO
#------------------------------------------------------------------------------------------------------

def calculate_fair_metrics(y_val, y_pred):
    """
    Summary:
        calculate p%-rule
    Arguments:
        y_val (pandas.DataFrame): true label
        y_pred (pandas.DataFrame): predicted  target
    Return:
        
    """
    
    cm=confusion_matrix(y_val, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    N = TP+FP+FN+TN #Total population
    # ACC = (TP+TN)/N #Accuracy
    TPR = TP/(TP+FN) # True positive rate
    FPR = FP/(FP+TN) # False positive rate
    FNR = FN/(TP+FN) # False negative rate
    PPP = (TP + FP)/N # % predicted as positive
    return np.array([TPR, FPR, FNR, PPP])


def fairness_metrics(y_val, y_pred, z_values):
    """
    Summary:
        Calculate fairness for subgroup of population
    Arguments:
        y_val (pandas.DataFrame): true label
        y_pred (pandas.DataFrame): predicted  target
        z_values (pandas.DataFrame): sensitive features
    Return:
        
    """
    
    y_val_0 = []
    y_val_1 = []
    y_pred_0 = []
    y_pred_1 = []
    for i  in range(len(z_values)):
        if z_values[i] == 1:
            y_val_1.append(y_val[i])
            y_pred_1.append(y_pred[i])
        else:
            y_val_0.append(y_val[i])
            y_pred_0.append(y_pred[i])

    # For class 0
    fm_0 = calculate_fair_metrics(y_val_0, y_pred_0)
    # For class 1
    fm_1 = calculate_fair_metrics(y_val_1, y_pred_1)

    fm_ratio = fm_0 / fm_1
    #print(fm_ratio)
    # res = "EOp:{:.2f} EO:{:.2f} DI:{:.2f}".format(fm_ratio[0], fm_ratio[1], fm_ratio[2], fm_ratio[3])

    # res ={'TPR': fm_ratio[0],
    #       'FPR': fm_ratio[1],
    #       'FNR': fm_ratio[2],
    #       'PPP': fm_ratio[3],
    #     }
    return  fm_ratio