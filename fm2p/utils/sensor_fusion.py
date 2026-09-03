# -*- coding: utf-8 -*-
"""
fm2p/utils/sensor_fusion.py

Kalman filter and IMU sensor fusion for head orientation estimation.

Adapted from github.com/wehr-lab/autopilot/tree/parallax.

Classes
-------
Kalman
    General-purpose Kalman filter with predict/update interface.
ImuOrientation
    Fuse accelerometer and gyroscope readings into roll and pitch estimates.


DMM, September 2025
"""

import numpy as np
from time import time


class Kalman():
    """ Kalman filter.

    From https://github.com/wehr-lab/autopilot/tree/parallax
    """

    def __init__(self, dim_state: int, dim_measurement: int = None, dim_control: int=0,
                 *args, **kwargs):

        self.dim_state = dim_state # type: int
        if dim_measurement is None:
            self.dim_measurement = self.dim_state # type: int
        else:
            self.dim_measurement = dim_measurement # type: int
        self.dim_control = dim_control # type: int

        self._init_arrays()

    def _init_arrays(self, state=None):
        """
        Initialize the arrays
        """
        # State arrays
        if state is not None:
            # TODO: check it's the right shape
            self.x_state = state
        else:
            self.x_state = np.zeros((self.dim_state, 1))

        # initialize kalman arrays
        self.P_cov               = np.eye(self.dim_state)                           # uncertainty covariance
        self.Q_proc_var          = np.eye(self.dim_state)                           # process uncertainty
        self.B_control           = np.eye(self.dim_control)                         # control transition matrix
        self.F_state_trans       = np.eye(self.dim_state)                           # x_state transition matrix
        if self.dim_state == self.dim_measurement:
            self.H_measure = np.eye(self.dim_measurement)
        else:
            self.H_measure           = np.zeros((self.dim_measurement, self.dim_state)) # measurement function
        self.R_measure_var       = np.eye(self.dim_measurement)                     # measurement uncertainty
        self._alpha_sq           = 1.                                               # fading memory control
        self.M_proc_measure_xcor = np.zeros((self.dim_state, self.dim_measurement)) # process-measurement cross correlation
        self.z_measure           = np.array([[None] * self.dim_measurement]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((self.dim_state, self.dim_measurement)) # kalman gain
        self.y = np.zeros((self.dim_measurement, 1))
        self.S = np.zeros((self.dim_measurement, self.dim_measurement)) # system uncertainty
        self.SI = np.zeros((self.dim_measurement, self.dim_measurement)) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(self.dim_state)

        # these will always be a copy of x_state,P_cov after predict() is called
        self.x_prior = self.x_state.copy()
        self.P_prior = self.P_cov.copy()

        # these will always be a copy of x_state,P_cov after update() is called
        self.x_post = self.x_state.copy()
        self.P_post = self.P_cov.copy()

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next x_state (prior) using the Kalman filter x_state propagation
        equations.

        Parameters
        ----------

        u : np.array, default 0
            Optional control vector.

        B : np.array(dim_state, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B_control`.

        F : np.array(dim_state, dim_state), or None
            Optional x_state transition matrix; a value of None
            will cause the filter to use `self.F_state_trans`.

        Q : np.array(dim_state, dim_state), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q_proc_var`.
        """
        if B is None:
            B = self.B_control
        if F is None:
            F = self.F_state_trans
        if Q is None:
            Q = self.Q_proc_var
        elif np.isscalar(Q):
            Q = np.eye(self.dim_state) * Q

        # x_state = Fx + Bu
        if B is not None and u is not None:
            # make sure control vector is column
            u = np.atleast_2d(u)
            if u.shape[1] > u.shape[0]:
                u = u.T
            self.x_state = np.dot(F, self.x_state) + np.dot(B, u)
        else:
            self.x_state = np.dot(F, self.x_state)

        # P_cov = FPF' + Q_proc_var
        self.P_cov = self._alpha_sq * np.dot(np.dot(F, self.P_cov), F.T) + Q

        # save prior
        np.copyto(self.x_prior, self.x_state)
        np.copyto(self.P_prior, self.P_cov)

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z_measure) to the Kalman filter.

        If z_measure is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z_measure is set to None.

        Parameters
        ----------
        z : (dim_measurement, 1): array_like
            measurement for this update. z_measure can be a scalar if dim_measurement is 1,
            otherwise it must be convertible to a column vector.

            If you pass in a value of H_measure, z_measure must be a column vector the
            of the correct size.

        R : np.array, scalar, or None
            Optionally provide R_measure_var to override the measurement noise for this
            one call, otherwise  self.R_measure_var will be used.

        H : np.array, or None
            Optionally provide H_measure to override the measurement function for this
            one call, otherwise self.H_measure will be used.
        """
        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z_measure = np.array([[None] * self.dim_measurement]).T
            np.copyto(self.x_post, self.x_state)
            np.copyto(self.P_post, self.P_cov)
            self.y = np.zeros((self.dim_measurement, 1))
            return

        if R is None:
            R = self.R_measure_var
        elif np.isscalar(R):
            R = np.eye(self.dim_measurement) * R

        if H is None:
            z = self._reshape_z(z, self.dim_measurement, self.x_state.ndim)
            H = self.H_measure

        # y = z_measure - Hx
        # error (residual) between measurement and prediction
        self.y = z - np.dot(H, self.x_state)

        # common subexpression for speed
        PHT = np.dot(self.P_cov, H.T)

        # S = HPH' + R_measure_var
        # project system uncertainty into measurement space
        self.S = np.dot(H, PHT) + R
        self.SI = np.linalg.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = np.dot(PHT, self.SI)

        # x_state = x_state + Ky
        # predict new x_state with residual scaled by the kalman gain
        self.x_state = self.x_state + np.dot(self.K, self.y)

        # P_cov = (I-KH)P_cov(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P_cov = (I-KH)P_cov usually seen in the literature.

        I_KH = self._I - np.dot(self.K, H)
        self.P_cov = np.dot(np.dot(I_KH, self.P_cov), I_KH.T) + np.dot(np.dot(self.K, R), self.K.T)

        # save measurement and posterior x_state
        np.copyto(self.z_measure, z)
        np.copyto(self.x_post, self.x_state)
        np.copyto(self.P_post, self.P_cov)
        return self.x_state

    def _reshape_z(self, z, dim_z, ndim):
        """ ensure z is a (dim_z, 1) shaped vector"""

        z = np.atleast_2d(z)
        if z.shape[1] == dim_z:
            z = z.T

        if z.shape != (dim_z, 1):
            raise ValueError('z (shape {}) must be convertible to shape ({}, 1)'.format(z.shape, dim_z))

        if ndim == 1:
            z = z[:, 0]

        if ndim == 0:
            z = z[0, 0]

        return z

    def process(self, z, **kwargs):
        """
        Call predict and update, passing the relevant kwargs

        Args:
            z ():
            **kwargs ():

        Returns:
            np.ndarray: self.x_state
        """

        # prepare args for predict and call
        predict_kwargs = {k:kwargs.get(k, None) for k in ("u", "B", "F", "Q")}
        self.predict(**predict_kwargs)

        # same thing for update
        update_kwargs = {k: kwargs.get(k, None) for k in ('R', 'H')}
        return self.update(z, **update_kwargs)

    def residual_of(self, z):
        """
        Returns the residual for the given measurement (z_measure). Does not alter
        the x_state of the filter.
        """
        return z - np.dot(self.H_measure, self.x_prior)

    def measurement_of_state(self, x):
        """
        Helper function that converts a x_state into a measurement.

        Parameters
        ----------

        x : np.array
            kalman x_state vector

        Returns
        -------

        z_measure : (dim_measurement, 1): array_like
            measurement for this update. z_measure can be a scalar if dim_measurement is 1,
            otherwise it must be convertible to a column vector.
        """

        return np.dot(self.H_measure, x)

    @property
    def alpha(self):
        """
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.
        """
        return self._alpha_sq**.5

    @alpha.setter
    def alpha(self, value):
        if not np.isscalar(value) or value < 1:
            raise ValueError('alpha must be a float greater than 1')

        self._alpha_sq = value**2


# Accelerometer-to-gyroscope crossover for tilt fusion, Hz.  Below this the
# accelerometer sets the estimate (gravity is a reliable absolute reference when
# the head is not accelerating); above it the gyroscope does (it is immune to
# linear acceleration but drifts, so it cannot be trusted at DC).
#
# Chosen by sweeping against synthetic ground truth built to look like this
# recording: 12 deg RMS pitch with a 1/f spectrum to 8 Hz, an accelerometer
# corrupted by linear acceleration correlated with head movement, and a gyro
# carrying white noise, a scale error and a bias.  RMSE against truth, in deg:
#
#     crossover   0.05  0.10  0.20  0.35  0.50  0.75  1.00  2.00
#     mean        1.19  0.92  1.00  1.33  1.70  2.33  2.94  5.17
#
# against 13.57 for the accelerometer alone.  Below ~0.1 Hz gyro bias starts to
# accumulate faster than the accelerometer can pull it back; above ~0.2 Hz the
# accelerometer's locomotion artefacts are increasingly let through.  0.15 sits
# in the flat basin between, so the choice does not hinge on the exact gyro bias
# assumed -- which is as well, because the real bias could not be measured: the
# gyro reads exactly 0 during still periods, so its noise floor is below the
# sensor's own quantisation step.
#
# 0.5 Hz, the textbook default for head-mounted tilt fusion, is nearly 2x worse
# here; do not restore it without re-running that sweep.
CROSSOVER_HZ = 0.15


class ImuOrientation():
    """ IMU sensor fusion

    From https://github.com/wehr-lab/autopilot/tree/parallax

    Compute absolute orientation (roll, pitch) from accelerometer and gyroscope measurements
    (eg from :class:`.hardware.i2c.I2C_9DOF` )

    Uses a :class:`.timeseries.Kalman` filter, and implements :cite:`patonisFusionMethodCombining2018a` to fuse
    the sensors

    Can be used with accelerometer data only, or with combined accelerometer/gyroscope data for
    greater accuracy

    Arguments:
        invert_gyro (bool): if the gyroscope's orientation is inverted from accelerometer measurement, multiply
            gyro readings by -1 before using
        use_kalman (bool): Whether to use kalman filtering (True, default), or return raw trigonometric
            transformation of accelerometer readings (if provided, gyroscope readings will be ignored)

    Attributes:
        kalman (:class:`.transform.timeseries.Kalman`): If ``use_kalman == True`` , the Kalman Filter.

    References:
        :cite:`patonisFusionMethodCombining2018a`
        :cite:`abyarjooImplementingSensorFusion2015`
    """

    def __init__(self, use_kalman:bool = True, invert_gyro:bool=False,
                 crossover_hz:float=None, dt_nominal:float=None,
                 *args, **kwargs):
        """
        crossover_hz : float
            Frequency at which the filter hands over from accelerometer to
            gyroscope.  Below it the accelerometer dominates and pins the
            estimate to gravity; above it the gyroscope dominates and rides
            through the linear acceleration that corrupts the accelerometer
            whenever the head accelerates.  This is the only meaningful tuning
            knob: the Kalman steady-state gain depends on Q and R only through
            their RATIO, so fixing R=1 and setting Q from the desired crossover
            captures the whole tuning in one interpretable number.

            Defaults to CROSSOVER_HZ.  Passing None for either argument leaves
            Q and R at the identity matrices the Kalman class ships with, which
            is a gain of ~0.62 -- effectively an unsmoothed accelerometer, with
            the gyroscope contributing almost nothing.
        dt_nominal : float
            Sample interval used to convert crossover_hz into a gain.  Supply
            the recording's real median interval; the IMU here runs at ~144 Hz,
            not the 60 Hz an eye-camera-rate guess would give.
        """

        self.invert_gyro = invert_gyro # type: bool
        self._last_update = None # type: typing.Optional[float]
        self._dt = 0 # type: float
        # preallocate orientation array for filtered values
        self.orientation = np.zeros((2), dtype=float) # type: np.ndarray
        # and for unfiltered values so they aren't ambiguous
        self._orientation = np.zeros((2), dtype=float)  # type: np.ndarray

        # Updates applied since construction.  Needed because the offline path
        # cannot use _last_update (which is wall-clock) to tell whether the
        # filter has history, and keying off _dt would misfire on a genuine
        # dt of 0.
        self._n_updates = 0  # type: int

        self.kalman = None # type: typing.Optional[Kalman]
        if use_kalman:
            self.kalman = Kalman(dim_state=2, dim_measurement=2, dim_control=2)  # type: typing.Optional[Kalman]
            if crossover_hz is None:
                crossover_hz = CROSSOVER_HZ
            if crossover_hz and dt_nominal:
                # For a random-walk state with process variance q and
                # measurement variance r, the steady-state gain is
                # K = P/(P+r) with P = (q + sqrt(q^2 + 4qr))/2, which inverts to
                # q = r K^2 / (1 - K).  A first-order complementary filter with
                # the same crossover has K = 2*pi*f_c*dt, so one number sets it.
                K = float(2.0 * np.pi * crossover_hz * dt_nominal)
                K = min(max(K, 1e-6), 0.99)
                r = 1.0
                q = r * K ** 2 / (1.0 - K)
                self.kalman.Q_proc_var = np.eye(2) * q
                self.kalman.R_measure_var = np.eye(2) * r
                self.crossover_hz = float(crossover_hz)
                self.kalman_gain_ss = K
            else:
                self.crossover_hz = None
                self.kalman_gain_ss = None

    def process(self, accelgyro, dt=None):
        """

        Args:
            accelgyro (tuple, :class:`numpy.ndarray`): tuple of (accelerometer[x,y,z], gyro[x,y,z]) readings as arrays, or
                an array of just accelerometer[x,y,z]
            dt (float, optional): seconds since the previous sample.  Supply this
                when processing a RECORDED file.  Without it the class times
                itself with the wall clock, which is correct for a live sensor
                but meaningless offline: replaying a file, successive calls are
                microseconds apart, so ``gyro * dt`` is ~0 and the gyroscope
                contributes nothing to the fusion.  Passing the real sample
                interval is what the TODO below asks for.

        Returns:
            :class:`numpy.ndarray`: filtered [roll, pitch] calculations in degrees
        """
        # check what we were given...
        if isinstance(accelgyro, (tuple, list)) and len(accelgyro) == 2:
            # combined accelerometer and gyroscope readings
            accel, gyro = accelgyro
        elif isinstance(accelgyro, np.ndarray) and np.squeeze(accelgyro).shape[0] == 3:
            # just accelerometer readings
            accel = accelgyro
            gyro = None
        else:
            # idk lol
            # self.logger.exception(f'Need input to be a tuple of accelerometer and gyroscope readings, or an array of accelerometer readings. got {accelgyro}')
            print('Error')
            return

        # convert accelerometer readings to roll and pitch
        pitch = 180*np.arctan2(accel[0], np.sqrt(accel[1]**2 + accel[2]**2))/np.pi
        roll = 180*np.arctan2(accel[1], np.sqrt(accel[0]**2 + accel[2]**2))/np.pi

        if self.kalman is None:
            # store orientations in external attribute if not using kalman filter
            self.orientation[:] = (roll, pitch)
            return self.orientation.copy()
        else:
            # if using kalman filter, use private array to store raw orientation
            self._orientation[:] = (roll, pitch)

        # dt is either supplied by the caller (offline, from the recording's own
        # timestamps) or measured against the wall clock (live sensor).
        first = (self._last_update is None if dt is None
                 else self._n_updates == 0)

        if first or gyro is None:
            # first time through don't have dt to scale gyro by
            self.orientation[:] = np.squeeze(self.kalman.process(self._orientation))
            if dt is None:
                self._last_update = time()
            else:
                self._dt = float(dt)
        else:
            if self.invert_gyro:
                gyro = np.asarray(gyro, dtype=float) * -1

            if dt is None:
                # get dt for time since last update
                update_time = time()
                self._dt = update_time-self._last_update
                self._last_update = update_time
            else:
                self._dt = float(dt)

            if self._dt>1:
                # if it's been really long, the gyro read is pretty much useless and will give ridiculous reads
                self.orientation[:] = np.squeeze(self.kalman.process(self._orientation))
            else:
                # run predict and update stages separately to incorporate gyro
                self.kalman.predict(u=gyro[0:2]*self._dt)
                self.orientation[:] = np.squeeze(self.kalman.update(self._orientation))

        self._n_updates += 1
        return self.orientation.copy()

