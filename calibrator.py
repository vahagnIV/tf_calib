import numpy as np
import tensorflow as tf

try:
    Session = tf.compat.v1.Session
except ImportError:
    Session = tf.Session

from camera import Camera
from distortion import Radial


class Calibrator:
    def __init__(self, number_of_cameras: int):
        self.cameras = [Camera(Radial()) for i in range(number_of_cameras)]
        self.session = Session()
        self.session.run([tf.global_variables_initializer()])

    def train(self, c: np.ndarray, xi: np.ndarray):
        metrics = []
        for camera in self.cameras:
            metrics.append(('loss', camera.train(xi, c, self.session)))
        return metrics

    def get_intristics(self):
        return [cam.get_intrinsic_matrix(self.session) for cam in self.cameras]

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()
