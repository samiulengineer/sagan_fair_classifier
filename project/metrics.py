
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from sklearn.metrics import accuracy_score, roc_auc_score
from config import config, initializing
initializing()


def p_rule(y_pred, z_values, threshold=config["pRuleThreshold"]):
    y_z_1 = y_pred[z_values ==
                   1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values ==
                   0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1/odds]) * 100


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
        # 'f1_score': sm.metrics.f1_score,
        # 'precision': sm.metrics.precision,
        # 'recall': sm.metrics.recall,
        # 'accuracy': tf.keras.metrics.Accuracy(),
        'AUC': tf.keras.metrics.AUC()
    }
