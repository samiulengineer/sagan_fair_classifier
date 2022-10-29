
import keras.backend as K
import numpy as np
import tensorflow as tf

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


def p_rule(y_pred, z_values, threshold=config["pRuleThreshold"]):
    y_z_1 = y_pred[z_values ==
                   1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values ==
                   0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1/odds]) * 100


def fairness_metrics(y_true, y_pred):

    # TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # TP = tf.keras.metrics.TruePositives()
    # TP.update_state(y_pred, y_true)
    # FP = tf.keras.metrics.FalsePositives()
    # FP.update_state(y_true, y_pred)
    # TN = tf.keras.metrics.TrueNegatives()
    # TN.update_state(y_true, y_pred)
    # FN = tf.keras.metrics.FalseNegatives()
    # FN.update_state(y_true, y_pred)

    # TP = TP.result().numpy()
    # FP = FP.result().numpy()
    # TN = TN.result().numpy()
    # FN = FN.result().numpy()

    # N = TP + FP + FN + TN

    # # Overall accuracy
    # ACC = (TP+TN)/N
    # # True positive rate
    # TPR = TP / (TP + FN)
    # # False positive rate
    # FPR = FP / (FP + TN)

    # # False negative rate
    # FNR = FN / (TP + FN)

    # # Percentage predicted as positive
    # PPP = (TP + FP) / N

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    return tp  # np.array([ACC, TPR, FPR, FNR, PPP])


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
        'fairness_metrics': BinaryTruePositives()
    }


# m = fairness_metrics([0, 1, 1, 1], [1, 0, 1, 1])
# m = tf.keras.metrics.TruePositives()
# m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
# print(m.result().numpy())
# print(m)
