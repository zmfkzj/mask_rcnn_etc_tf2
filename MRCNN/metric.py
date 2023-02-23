import tensorflow_addons as tfa
import tensorflow as tf


class F1Score(tfa.metrics.F1Score):
    def __init__(self, num_classes, average: str = None, threshold = None, name: str = "f1_score", dtype = None):
        super().__init__(num_classes, average, threshold, name, dtype)

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros", dtype=self.dtype, synchronization=tf.VariableSynchronization.AUTO
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")
