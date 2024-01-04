from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_model_analysis as tfma
import segmentation_models as sm

# Setting framework

sm.set_framework('tf.keras')
sm.framework()

# Keras MeanIoU
# ----------------------------------------------------------------------------------------------

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    '''
    Summary:
        MyMeanIOU inherit tf.keras.metrics.MeanIoU class and modifies update_state function.
        Computes the mean intersection over union metric.
        iou = true_positives / (true_positives + false_positives + false_negatives)
    Arguments:
        num_classes (int): The possible number of labels the prediction task can have
    Return:
        Class objects
    '''

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=3), tf.argmax(y_pred, axis=3), sample_weight)


# Metrics
# ----------------------------------------------------------------------------------------------

def get_metrics(config):
    """
    Summary:
        create keras MeanIoU object and all custom metrics dictornary
    Arguments:
        config (dict): configuration dictionary
    Return:
        metrics directories
    """

    m = MyMeanIOU(config['num_classes'])
    return {
        'my_mean_iou': m,
        'f1-score': tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.9),
        'precision': sm.metrics.precision,
        'recall': sm.metrics.recall
    }
