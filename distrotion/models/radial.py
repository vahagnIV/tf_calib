import tensorflow as tf
from distrotion import IDistortion


class Radial(IDistortion):
    def __init__(self):
        self.k1 = tf.Variable(0., dtype=tf.float64)
        self.k2 = tf.Variable(0., dtype=tf.float64)
        self.k3 = tf.Variable(0., dtype=tf.float64)

    def apply(self, xi_input: tf.Tensor, c_x: tf.Tensor, c_y: tf.Tensor) -> tf.Tensor:
        center = tf.stack([c_x, c_y])
        r2 = tf.reduce_sum(tf.squared_difference(xi_input, center) , 1)
        r4 = r2 ** 2
        r6 = r4 ** 2

        # All coefficients should have the same order
        coeff = (1 + self.k1 * r2 / 1e6 + self.k2 * r4 / 1e12 + self.k3 * r6 / 1e24)

        return tf.expand_dims(coeff, 1) * xi_input

    def get_optimizers(self, loss: tf.Tensor):
        return [tf.train.AdamOptimizer(1e-2).minimize(loss,
                                                      var_list=[self.k1, self.k2, self.k3])]

    def get_variables(self, session):
        return session.run([self.k1, self.k2, self.k3])
