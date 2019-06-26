import tensorflow as tf
from typing import List

class IDistortion:
    def apply(self, xi_input: tf.Tensor, c_x: tf.Tensor, c_y: tf.Tensor) -> tf.Tensor:
        return xi_input

    def get_optimizers(self, loss: tf.Tensor):
        return []

    def get_coefficients(self, session: tf.Session):
        return []

    def set_coefficients(self, session: tf.Session, coefficients ):
        pass
