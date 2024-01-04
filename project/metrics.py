from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_model_analysis as tfma
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=3), tf.argmax(y_pred, axis=3), sample_weight)
        
def get_metrics(config):
    m = MyMeanIOU(config['num_classes'])
    return {
        'my_mean_iou': m,
        'f1-score': tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.9),
        'precision': sm.metrics.precision,
        'recall': sm.metrics.recall
    }
