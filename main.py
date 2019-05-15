import cv2
import os
import glob
from calibrator import Calibrator
import numpy as np

try:
    import python.modules.tf_calib
except ImportError:
    pass


def calibrate(path: str, filter: str, nrows: int, ncols: int):
    calibrator = Calibrator(1)
    objp = np.zeros((nrows * ncols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nrows, 0:ncols].T.reshape(-1, 2)
    counter = 0
    image_names = glob.glob(os.path.join(path, filter))
    print(image_names)

    for i in range(100):
        for imname in glob.glob(os.path.join(path, filter)):
            counter += 1
            image = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
            f, corners = cv2.findChessboardCorners(image, (nrows, ncols), None)
            cv2.cornerSubPix(image, corners, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            print(counter)
            calibrator.train(objp, np.squeeze(corners.astype(np.float64)))
            # if counter == 300:
            #     for camera in calibrator.cameras:
            #         print(camera.get_intrinsic_matrix(calibrator.session))
            #         print(camera.distortion.get_variables(calibrator.session))
            #         return


if __name__ == '__main__':
    calibrate('data', '*.png', 8, 13)
