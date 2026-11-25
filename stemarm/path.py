"""Trajectory motion of the boom tip along a square path (Sections 3.2 and 4.6).

The command is generated in four steps:
  1. geometric path gamma(s) (here a square) and its original time law s(t):
     a rest-to-rest trapezoid on every edge and a rest of T_h at every corner;
  2. optional path-progress shaping S(t) = sum_i A_i s((t - t_i)/k) with the
     compression k = (T - t_acc)/T, Eqs. (39)-(45);
  3. bounded inverse kinematics q = (q1, q2, L) of every sample of gamma(S(t)),
     with the rigid forward kinematics FK(q) of Eq. (46);
  4. a cubic Hermite spline through the joint samples, evaluated by the solver.
An unshaped approach from the initial pose to the first corner precedes the path.
"""
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares

from .model import kinematics, rigid_tip
from .shaping import shape_path_progress

# Square of Section 4.6 (all lengths in m, times in s)
CENTER = np.array([-1.0, 0.0, 2.0])
SIDE = 1.0
EDGE_VMAX, EDGE_AMAX = 0.5, 1.0
CORNER_HOLD = 0.3
DT = 0.02                    # command sampling
APPROACH_TIME = 2.0
MIN_SAMPLES_PER_EDGE = 15
Q_START_DEG = (10.0, 0.0)    # initial joint angles
L_START = 0.05               # initial deployed length (ANCF elements need L > 0)
L_MIN, L_MAX, L_MARGIN = 0.05, 10.0, 0.02
Q2_PATH_MAX = 1.6            # rad, upper limit of joint 2
T_ACC_MARGIN = DT            # zero-time-delay scheme: t_acc = t_N + one command sample


def square_corners():
    """Closed corner sequence, counter-clockwise from the corner (x, y) = (-0.5, -0.5)."""
    h = 0.5 * SIDE
    local = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    order = [1, 2, 3, 0, 1]
    return np.column_stack((CENTER[0] + local[order, 0], CENTER[1] + local[order, 1], np.full(5, CENTER[2])))


class SquarePath:
    """Polyline gamma(s) with the normalised arc length s in [0, 1]."""

    def __init__(self):
        self.points = square_corners()
        edge_length = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(edge_length)))
        self.length = float(cumulative[-1])
        self.progress = cumulative / self.length
        self.progress[-1] = 1.0
        self.edge_length = edge_length

    def point(self, s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        out = np.column_stack([np.interp(s.reshape(-1), self.progress, self.points[:, a]) for a in range(3)])
        return out.reshape(s.shape + (3,))

    def distance(self, xyz):
        """Distance of points from the polyline."""
        xyz = np.atleast_2d(xyz)
        a, b = self.points[:-1], self.points[1:]
        ab = b - a
        ap = xyz[:, None, :] - a
        u = np.clip(np.einsum("nkj,kj->nk", ap, ab) / np.einsum("kj,kj->k", ab, ab), 0.0, 1.0)
        return np.min(np.linalg.norm(ap - u[..., None] * ab, axis=2), axis=1)


def edge_timing(distance, vmax=EDGE_VMAX, amax=EDGE_AMAX):
    """Duration, acceleration time and cruise time of a rest-to-rest edge."""
    t_acc = vmax / amax
    if amax * t_acc * t_acc >= distance:
        t_acc, t_cruise = np.sqrt(distance / amax), 0.0
    else:
        t_cruise = (distance - amax * t_acc * t_acc) / vmax
    return float(2.0 * t_acc + t_cruise), float(t_acc), float(t_cruise)


def edge_distance(t, distance, vmax=EDGE_VMAX, amax=EDGE_AMAX):
    total, t_acc, t_cruise = edge_timing(distance, vmax, amax)
    t = float(np.clip(t, 0.0, total))
    if t <= t_acc:
        return 0.5 * amax * t * t
    if t <= t_acc + t_cruise:
        return 0.5 * amax * t_acc * t_acc + amax * t_acc * (t - t_acc)
    t_remain = total - t
    return float(distance - 0.5 * amax * t_remain * t_remain)


def time_law(path, dt=DT, hold=CORNER_HOLD):
    """Original progress s(t) with its Cartesian samples (edges, then a corner rest)."""
    times, progress, positions, elapsed = [0.0], [0.0], [path.points[0].copy()], 0.0
    for i in range(len(path.points) - 1):
        d = float(path.edge_length[i])
        move = edge_timing(d)[0]
        local_t = np.linspace(0.0, move, max(MIN_SAMPLES_PER_EDGE, int(np.ceil(move / dt)) + 1))[1:]
        travelled = np.array([edge_distance(v, d) for v in local_t])
        s = path.progress[i] + (path.progress[i + 1] - path.progress[i]) * travelled / d
        xyz = path.points[i][None, :] + travelled[:, None] * ((path.points[i + 1] - path.points[i]) / d)[None, :]
        s[-1], xyz[-1] = path.progress[i + 1], path.points[i + 1]
        times.extend(elapsed + local_t)
        progress.extend(s)
        positions.extend(xyz)
        elapsed += move
        for t_hold in np.arange(dt, hold + 0.5 * dt, dt):
            times.append(elapsed + min(float(t_hold), hold))
            progress.append(path.progress[i + 1])
            positions.append(path.points[i + 1].copy())
        elapsed += hold
    return np.asarray(times), np.asarray(progress), np.asarray(positions)


class SquareCommand:
    """Joint command (q1, q2) and deployed length L along the square path.

    shaper: None for the unshaped command, else (A, T) of the shaper.
    zero_delay: compress the time law by k with t_acc = t_N + T_ACC_MARGIN,
    otherwise t_acc = 0 (ordinary shaper delay).
    corner_hold: rest T_h at every corner before compression [s].
    """

    def __init__(self, shaper=None, zero_delay=True, corner_hold=CORNER_HOLD):
        self.robot, self.root_local, self.r_mount, arm = kinematics()
        self.q_limits = [link["limits"] for link in arm["chain"]]
        self.path = SquarePath()
        self.fk = lambda q1, q2, length: rigid_tip(self.robot, self.root_local, self.r_mount, [q1, q2], length)

        times, progress, square_xyz = time_law(self.path, hold=corner_hold)
        self.path_time = float(times[-1])
        q_init = np.array([np.deg2rad(Q_START_DEG[0]), np.deg2rad(Q_START_DEG[1]), L_START])
        tip0 = self.fk(*q_init)
        # quintic straight-line approach from the initial tip to the first corner
        t_app = np.linspace(0.0, APPROACH_TIME, max(2, int(np.ceil(APPROACH_TIME / DT)) + 1))
        u = t_app / APPROACH_TIME
        sigma = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        approach_xyz = tip0[None, :] + sigma[:, None] * (square_xyz[0] - tip0)[None, :]
        self.n_approach = len(t_app)
        t_all = np.concatenate((t_app, APPROACH_TIME + times[1:]))
        xyz_all = np.vstack((approach_xyz, square_xyz[1:]))

        q1_lim, q2_lim = self.q_limits
        self.bounds_approach = (np.array([q1_lim[0], q2_lim[0], L_MIN]), np.array([q1_lim[1], q2_lim[1], L_MAX]))
        self.bounds_path = (np.array([q1_lim[0], max(q2_lim[0], 0.0), L_MIN + L_MARGIN]),
                            np.array([q1_lim[1], min(q2_lim[1], np.deg2rad(np.rad2deg(Q2_PATH_MAX))),
                                      L_MAX - L_MARGIN]))
        q, err = self._inverse_kinematics(xyz_all, q_init)
        self._check_continuity(q)

        self.shaper_duration, self.k, self.t_acc = 0.0, 1.0, 0.0
        progress_all = np.concatenate((np.zeros(self.n_approach - 1), progress))
        if shaper is not None:
            t_N = float(np.max(shaper[1]))
            self.shaper_duration = t_N
            self.t_acc = t_N + T_ACC_MARGIN if zero_delay else 0.0
            t_shaped, s_shaped, self.k = shape_path_progress(times, progress, shaper, DT, self.t_acc)
            xyz_shaped = self.path.point(s_shaped)
            # seeds: the unshaped joint table interpolated in s; refined where needed
            base = q[self.n_approach - 1:]
            s_unique, index = np.unique(progress, return_index=True)
            q_s = np.column_stack([np.interp(s_shaped, s_unique, base[index, a]) for a in range(3)])
            q_s = np.clip(q_s, *self.bounds_path)
            err_s = np.array([np.linalg.norm(self.fk(*qq) - p) for qq, p in zip(q_s, xyz_shaped)])
            for row in np.flatnonzero(err_s > 1.0e-6):
                sol = least_squares(lambda v: self.fk(*v) - xyz_shaped[row], q_s[row], bounds=self.bounds_path,
                                    xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=20)
                q_s[row], err_s[row] = sol.x, np.linalg.norm(self.fk(*sol.x) - xyz_shaped[row])
            self._check_continuity(q_s)
            t_all = np.concatenate((t_app, APPROACH_TIME + t_shaped[1:]))
            q = np.vstack((q[:self.n_approach], q_s[1:]))
            xyz_all = np.vstack((xyz_all[:self.n_approach], xyz_shaped[1:]))
            err = np.concatenate((err[:self.n_approach], err_s[1:]))
            progress_all = np.concatenate((np.zeros(len(t_app)), s_shaped[1:]))
        if np.max(err) > 2.0e-3:
            raise RuntimeError(f"path outside the reachable workspace: IK error {np.max(err):.4f} m")

        # joint rates by finite differences; rest at the start, the path start and the end
        dq = np.gradient(q, t_all, axis=0, edge_order=2)
        rest = {0, self.n_approach - 1, len(dq) - 1}
        if shaper is None:
            for k in range(1, len(xyz_all) - 1):
                v0, v1 = xyz_all[k] - xyz_all[k - 1], xyz_all[k + 1] - xyz_all[k]
                n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
                if n0 > 1e-12 and n1 > 1e-12 and np.dot(v0, v1) / (n0 * n1) < 0.95:
                    rest.add(k)
        else:
            for corner in self.path.progress[1:]:
                k = int(np.argmin(np.abs(progress_all - corner)))
                if abs(progress_all[k] - corner) < 2.0e-4:
                    rest.add(k)
        for k in rest:
            dq[k] = 0.0
        self.times, self.q_table, self.target, self.ik_error = t_all, q, xyz_all, err
        self.splines = [CubicHermiteSpline(t_all, q[:, a], dq[:, a]) for a in range(3)]
        self.end_time = float(t_all[-1])
        self.q_start = np.array([np.deg2rad(Q_START_DEG[0]), np.deg2rad(Q_START_DEG[1])])

    def _inverse_kinematics(self, targets, seed):
        rows, errors = [], []
        for k, target in enumerate(targets):
            lower, upper = self.bounds_approach if k <= self.n_approach - 1 else self.bounds_path
            seed = np.minimum(np.maximum(seed, lower), upper)
            sol = least_squares(lambda v: self.fk(*v) - target, seed, bounds=(lower, upper),
                                xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=40)
            seed = sol.x.copy()
            rows.append(seed.copy())
            errors.append(float(np.linalg.norm(self.fk(*seed) - target)))
        return np.asarray(rows), np.asarray(errors)

    @staticmethod
    def _check_continuity(q):
        if np.any(np.max(np.abs(np.diff(q, axis=0)), axis=0) > 0.5):
            raise RuntimeError("IK branch switch: the path probably passes the joint-1 axis")

    # ---- command interface used by the model ------------------------------
    def _pva(self, t):
        tt = float(np.clip(float(t), 0.0, self.end_time))
        value = np.array([float(s(tt)) for s in self.splines])
        if tt <= 0.0 or tt >= self.end_time:
            return value, np.zeros(3), np.zeros(3)
        return (value, np.array([float(s(tt, 1)) for s in self.splines]),
                np.array([float(s(tt, 2)) for s in self.splines]))

    def joint_pva(self, t):
        value, rate, acc = self._pva(t)
        q = value[:2].copy()
        q[1] = float(np.clip(q[1], *self.q_limits[1]))     # q1 may wind beyond +-pi, q2 is limited
        return q, rate[:2], acc[:2]

    def length_pva(self, t):
        value, rate, acc = self._pva(t)
        return float(value[2]), float(rate[2]), float(acc[2])

    def desired_tip(self, t):
        q, _, _ = self.joint_pva(t)
        return self.fk(q[0], q[1], self.length_pva(t)[0])

    def check(self, dt=0.002):
        """Peak joint rates and the contour error of the commanded rigid path, on the path only."""
        times = np.append(np.arange(0.0, self.end_time, dt), self.end_time)
        pva = [self._pva(v) for v in times]
        q = np.array([p[0] for p in pva])
        on_path = times >= APPROACH_TIME - 1e-12
        xyz = np.array([self.fk(*row) for row in q[on_path]])
        contour = 1e3 * self.path.distance(xyz)
        return {"qd_max": np.max(np.abs([p[1] for p in pva])[on_path], axis=0).tolist(),
                "qdd_max": np.max(np.abs([p[2] for p in pva])[on_path], axis=0).tolist(),
                "contour_rms_mm": float(np.sqrt(np.mean(contour**2))), "contour_max_mm": float(np.max(contour))}

    def corner_speed(self, dt=DT, radius=0.005):
        """Largest speed at which the commanded rigid path passes a corner (v_c of Table 3)."""
        times = np.append(np.arange(0.0, self.end_time, dt), self.end_time)
        q = np.array([self._pva(v)[0] for v in times])
        xyz = np.array([self.fk(*row) for row in q])
        keep = times >= APPROACH_TIME
        t, x, y = times[keep], xyz[keep, 0], xyz[keep, 1]
        v = np.hypot(np.gradient(x, t), np.gradient(y, t))
        speeds = []
        for corner in self.path.points[:4]:
            near = np.hypot(x - corner[0], y - corner[1]) < radius
            speeds.append(v[near].min() if near.any() else np.nan)
        return float(np.nanmax(speeds))
