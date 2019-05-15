import tensorflow as tf


def create_rotation_matrix(angles: tf.Tensor):
    """
    Generates a rotation matrix from 3 Bryiant angles
    :param angles: the placeholders for the 3 Bryant angles of shape (3, ?)
    :return: The rotation matrix
    """
    a1 = tf.reshape(angles[0], (-1,))
    a2 = tf.reshape(angles[1], (-1,))
    a3 = tf.reshape(angles[2], (-1,))

    c1 = tf.cos(a1)
    c2 = tf.cos(a2)
    c3 = tf.cos(a3)
    s1 = tf.sin(a1)
    s2 = tf.sin(a2)
    s3 = tf.sin(a3)

    r11 = c2 * c3
    r12 = - c2 * s3
    r13 = s2
    r21 = c1 * s3 + c3 * s1 * s2
    r22 = c1 * c3 - s1 * s2 * s3
    r23 = -c2 * s1
    r31 = s1 * s3 - c1 * s2 * c3
    r32 = s1 * c3 + c1 * s2 * s3
    r33 = c1 * c2

    first_row = tf.stack([r11, r12, r13])
    second_row = tf.stack([r21, r22, r23])
    third_row = tf.stack([r31, r32, r33])
    rotation_matrix = tf.stack([first_row, second_row, third_row])
    return tf.reshape(rotation_matrix, (3, 3, -1))
