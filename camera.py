import numpy as np
import tensorflow as tf
from utils import create_rotation_matrix
from distrotion import IDistortion
from scipy.optimize import newton_krylov
from scipy.optimize.nonlin import NoConvergence

class Camera:
    def __init__(self, distortion: IDistortion = IDistortion()):
        CAMERA_PARAMETER_NORMALIZATION_CONSTANT = 1000

        self.angles = tf.placeholder(tf.float64, (3, None))
        self.R = create_rotation_matrix(self.angles)
        self.xi_input = tf.placeholder(tf.float64, (None, 2))
        self.c = tf.placeholder(tf.float64, (None, 3))

        # We use normalized parameters  fx_real / 1000
        self.f_x = tf.Variable(1.52137, dtype=tf.float64)
        self.f_y = tf.Variable(1.5195, dtype=tf.float64)
        self.c_x = tf.Variable(0.6984, dtype=tf.float64)
        self.c_y = tf.Variable(0.442494, dtype=tf.float64)

        self.distortion = distortion
        self.xi = self.distortion.apply(self.xi_input, self.c_x * CAMERA_PARAMETER_NORMALIZATION_CONSTANT,
                                        self.c_y * CAMERA_PARAMETER_NORMALIZATION_CONSTANT)

        first_row = tf.stack([self.f_x, 0, self.c_x])
        second_row = tf.stack([0, self.f_y, self.c_y])
        third_row = tf.constant([0, 0, 1 / CAMERA_PARAMETER_NORMALIZATION_CONSTANT], dtype=tf.float64)

        # ============== intrinsic matrix =====================
        self.K = CAMERA_PARAMETER_NORMALIZATION_CONSTANT * tf.stack([first_row, second_row, third_row])
        self.K_inverse = tf.matrix_inverse(self.K)

        u, u_inverse = self._init_u(self.xi)

        Omegai = self._init_Omega(u, u_inverse)

        Ai = self._init_Ai(self.R, self.c, Omegai)

        Omega = tf.reduce_sum(Omegai, axis=2)
        W = tf.matrix_inverse(Omega)

        U = self._init_U(W, tf.shape(u)[1])

        self.M = tf.tensordot(Ai, U, [[0, 1], [0, 2]])
        Omegajcj = tf.expand_dims(Omegai, 3) * self.c
        self.M = tf.tensordot(self.M, Omegajcj, [[1, 2], [0, 2]])
        self.RTM = tf.expand_dims(tf.transpose(self.R, [1, 2, 0]), 3) * self.M
        self.RTM = tf.reduce_sum(self.RTM, 2)

        self.loss = tf.tensordot(Ai, U, [[0, 1], [0, 2]])
        self.loss *= tf.transpose(Ai, [2, 0, 1])
        self.loss = tf.reduce_sum(self.loss, axis=[1, 2]) / tf.cast(tf.shape(u)[1], tf.float64)

        self.s, self.x, self.RcpT = self._get_visalization_parameters(Ai, W, u, u_inverse)

        loss = tf.reduce_sum(self.loss)
        self.optimizers = [
                              tf.train.AdamOptimizer(5e-2).minimize(loss, var_list=[self.f_x, self.f_y]),
                              tf.train.AdamOptimizer(1e-1).minimize(loss, var_list=[self.c_x, self.c_y])
                          ] + self.distortion.get_optimizers(loss)

    def _init_u(self, xi):
        ones = tf.ones((tf.shape(xi)[0], 1), dtype=tf.float64)

        u = tf.concat([xi, ones], axis=1)

        u = tf.tensordot(self.K_inverse, u, [1, 1])

        u_inverse = u / tf.reduce_sum(u ** 2, axis=0)

        return u, u_inverse

    def _init_Omega(self, u: tf.Tensor, u_inverse: tf.Tensor) -> tf.Tensor:
        delta = tf.eye(3, dtype=u.dtype)
        delta = tf.reshape(delta, (-1, 1))
        delta = tf.tile(delta, [1, tf.shape(u)[1]])
        delta = tf.reshape(delta, (3, 3, -1))

        Omegai = delta - tf.transpose(tf.expand_dims(u_inverse, 2) * tf.transpose(u, [1, 0]),
                                      [0, 2, 1])

        return Omegai

    def _init_Ai(self, R: tf.Tensor, c: tf.Tensor, Omegai: tf.Tensor):
        Rci = tf.transpose(tf.tensordot(R, c, [1, 1]), [0, 2, 1])

        Ai = tf.expand_dims(Omegai, 3) * Rci
        Ai = tf.reduce_sum(Ai, 1)
        return Ai

    def _init_U(self, W: tf.Tensor, N):
        WJ = tf.expand_dims(W, 2)
        WJ = tf.expand_dims(WJ, 3)
        WJ = tf.tile(WJ, [1, 1, N, N])

        II = tf.eye(3, dtype=tf.float64)
        II = tf.expand_dims(II, 2)
        II = tf.expand_dims(II, 3)
        II = II * tf.eye(N, dtype=tf.float64)

        return -WJ + II

    def _get_visalization_parameters(self, Ai: tf.Tensor, W, u, u_inverse):
        A = tf.reduce_sum(Ai, axis=1)
        T = tf.tensordot(W, A, [1, 0])

        T_tiled = tf.expand_dims(T, 2)
        T_tiled = tf.tile(T_tiled, (1, 1, tf.shape(u_inverse)[1]))

        Rc = tf.tensordot(self.R, self.c, [1, 1])
        RcpT = Rc - T_tiled

        uR_uu = tf.tensordot(u_inverse, self.R, [0, 0])
        uRc_uu = tf.expand_dims(self.c, 2) * uR_uu
        uiRci_uu = tf.reduce_sum(uRc_uu, 1)
        uT_uu = tf.tensordot(u_inverse, T, [0, 0])
        s = uiRci_uu - uT_uu

        x = tf.expand_dims(u, 2) * s

        return s, x, RcpT

    def _find_rotation_matrix_guess(self, xi: np.ndarray, c: np.ndarray, session: tf.Session):
        N = 30

        intervals = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], dtype=np.float64)
        grid1 = np.linspace(intervals[0, 0], intervals[0, 1], N)
        grid2 = np.linspace(intervals[1, 0], intervals[1, 1], N)
        grid3 = np.linspace(intervals[2, 0], intervals[2, 1], N)
        meshgrid = np.array(np.meshgrid(grid1, grid2, grid3)).reshape((3, -1))

        loss, s = session.run([self.loss, self.s], feed_dict={
            self.xi_input: xi, self.c: c, self.angles: meshgrid
        })
        indices = np.argsort(loss)

        for ind in indices:
            if all(s[:, ind] > 0):
                break
        # print(loss[ind])
        return meshgrid[:, ind, np.newaxis]

    def find_best_rotation_parameters(self, xi: np.ndarray, c: np.ndarray, session: tf.Session):
        guess = self._find_rotation_matrix_guess(xi, c, session)

        def loss_function(angles: np.ndarray):
            mr = session.run([self.RTM], feed_dict={self.xi_input: xi, self.c: c, self.angles: angles})[0]

            mr = mr[:, 0, :]

            return np.array([mr[0, 1] - mr[1, 0], mr[0, 2] - mr[2, 0], mr[2, 1] - mr[1, 2]])
        try:
            sol = newton_krylov(loss_function, guess, method='lgmres', verbose=0)
        except NoConvergence:
            return None
        return sol

    def get_intrinsic_matrix(self, session: tf.Session):
        return session.run([self.K])[0]


    def train(self, xi: np.ndarray, c: np.ndarray, session: tf.Session):
        angles = self.find_best_rotation_parameters(xi, c, session)
        if angles is None:
            return None
        # print(angles)

        b = session.run(self.optimizers + [self.x, self.RcpT, self.loss],
                              feed_dict={self.xi_input: xi, self.c: c, self.angles: angles})

        # rcpt = b[-2]
        # x = b[-3]
        #
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.invert_yaxis()
        # ax.invert_zaxis()
        # ax.set_xlabel('$X$', fontsize=20, rotation=150)
        # ax.set_ylabel('$Y$')
        # ax.set_zlabel('$Z$')
        # ax.scatter(xs=rcpt[0, 0, :], zs=rcpt[2, 0, :], ys=rcpt[1, 0, :], zdir='z', s=20, c=None, depthshade=True)
        # ax.scatter(xs=rcpt[0, 0, 0:1], zs=rcpt[2, 0, 0:1], ys=rcpt[1, 0, 0:1], zdir='z', s=20, c=None, depthshade=True)
        # ax.scatter(xs=x[0, :, 0], zs=x[2, :, 0], ys=x[1, :, 0], zdir='z', s=20, c=None, depthshade=True)
        # fig.show()
        return b[-1]

        pass
