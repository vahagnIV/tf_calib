import tensorflow as tf


class IDistortion:
    def apply(self, xi_input: tf.Tensor, c_x: tf.Tensor, c_y: tf.Tensor) -> tf.Tensor:
        return xi_input

    def get_optimizers(self, loss: tf.Tensor):
        return []
