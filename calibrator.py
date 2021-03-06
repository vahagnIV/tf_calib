import numpy as np
import tensorflow as tf
import sys
import threading
from typing import List

try:
    Session = tf.compat.v1.Session
except ImportError:
    Session = tf.Session

from camera import Camera
from distortion import Radial


class Calibrator:
    def __init__(self, number_of_cameras: int):
        print("Calibrator Initializing")
        print(Camera)
        print(number_of_cameras)
        print('Threadid = ', threading.current_thread().ident)
        self.session = Session()
        print('Session created')

        print('Loaded modules', sys.modules.keys())
        self.cameras = [Camera(Radial()) for i in range(number_of_cameras)]
        print('Cameras created')
        self.session.run([tf.global_variables_initializer()])

    def train(self, c: np.ndarray, xi: List):
        metrics = []
        for index, camera in enumerate(self.cameras):
            metrics.append([camera.train(xi[index], c, self.session)])
        return (metrics, None)

    def get_intrinsics(self):
        return [(cam.get_intrinsic_matrix(self.session), cam.distortion.get_coefficients(self.session)) for cam in self.cameras]

    def set_intrinsics(self, intrinsics):
        print(intrinsics)
        for cam, params in zip(self.cameras, intrinsics):
            cam.set_intrinsic_matrix(self.session, params[0])
            cam.distortion.set_coefficients(self.session, params[1])

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()
        print("Calibrator disposed")
