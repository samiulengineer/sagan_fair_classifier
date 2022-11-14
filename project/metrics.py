
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from config import config, initializing
initializing()


class BinaryTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives



# P% rule metrics
#------------------------------------------------------------------------------------------------------

def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1/odds]) * 100


# Fairness metrics for DI, EOp, EO
#------------------------------------------------------------------------------------------------------

def calculate_fair_metrics(y_val, y_pred):
    
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
    """Calculate fairness for subgroup of population"""
    # print(type(y_val))
    # print(type(y_pred))
    # print(type(z_values))
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
    # res = "TPR:{:.2f} FPR:{:.2f} FNR:{:.2f} PPP:{:.2f}".format(fm_ratio[0], fm_ratio[1], fm_ratio[2], fm_ratio[3])

    # res ={'TPR': fm_ratio[0],
    #       'FPR': fm_ratio[1],
    #       'FNR': fm_ratio[2],
    #       'PPP': fm_ratio[3],
    #     }
    return  fm_ratio



def get_metrics():
    """
    Summary:
        create keras MeanIoU object and all custom metrics dictornary
    Arguments:
        config (dict): configuration dictionary
    Return:
        metrics directories
    """

    return {
        'TP': tf.keras.metrics.TruePositives(),
        # 'FP': tf.keras.metrics.FalsePositives(),
        # 'TN': tf.keras.metrics.TrueNegatives(),
        # 'FN': tf.keras.metrics.FalseNegatives(),
        # 'AUC': tf.keras.metrics.AUC(),
        'fairness_metrics': fairness_metrics
    }


# m = fairness_metrics([0, 1, 1, 1], [1, 0, 1, 1])
# m = tf.keras.metrics.TruePositives()
# m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
# print(m.result().numpy())
# print(m)
