import numpy as np
import tensorflow as tf
from camera import Camera
from distrotion import Radial


class Calibrator:
    def __init__(self, number_of_cameras: int):
        self.cameras = [Camera(Radial()) for i in range(number_of_cameras)]
        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer()])
        pass

    def train(self, c: np.ndarray, xi: np.ndarray):
        for camera in self.cameras:
            loss = camera.train(xi, c, self.session)
            print(loss)
            # print(camera.get_intrinsic_matrix(session=self.session))
            # print(camera.distortion.get_variables(session=self.session))
        pass
