"""Commanded motions q(t) = [q1, q2] and L(t) passed to the model.

Each command provides
    joint_pva(t)  -> (q, qd, qdd), arrays of the two joint angles [rad]
    length_pva(t) -> (L, Ld, Ldd), deployed length [m]
    end_time      -> time at which the (shaped) command comes to rest [s]
"""
import numpy as np

from . import params as P
from .shaping import Trapezoid, length_trapezoid_pva, length_trapezoid_time


class PointToPoint:
    """Synchronised point-to-point slew, optionally with a boom extension (Section 3.2).

    Each moving axis follows a trapezoidal profile; all axes are stretched to
    the duration of the slowest one so that they start and stop together. The
    shaper is applied to every component, Eq. (38).
    """

    def __init__(self, q_start_deg, q_end_deg, shaper=None, length=1.0, length_end=None,
                 length_vmax=None, length_amax=None, vmax=P.JOINT_VMAX, amax=P.JOINT_AMAX):
        self.q_start = np.deg2rad(np.asarray(q_start_deg, dtype=float))
        self.delta = np.deg2rad(np.asarray(q_end_deg, dtype=float)) - self.q_start
        self.moving = np.abs(self.delta) > 0.0
        self.profiles = [Trapezoid(d, vmax, amax) for d in self.delta]
        self.shaper = shaper if shaper is not None else (np.array([1.0]), np.array([0.0]))
        self.length_start = float(length)
        self.length_end = float(length if length_end is None else length_end)
        self.length_vmax, self.length_amax = length_vmax, length_amax
        self.extends = self.length_end != self.length_start

        durations = [p.total_time for p, m in zip(self.profiles, self.moving) if m]
        if self.extends:
            durations.append(length_trapezoid_time(self.length_start, self.length_end,
                                                   length_vmax, length_amax))
        # common duration T; never shorter than one host sample
        self.duration = max(durations + [P.HOST_TRAJ_DT])
        self.end_time = self.duration + float(np.max(self.shaper[1]))

    def joint_pva(self, t):
        q, qd, qdd = self.q_start.copy(), np.zeros(2), np.zeros(2)
        for j, (profile, moving) in enumerate(zip(self.profiles, self.moving)):
            if not moving:
                continue
            pos = vel = acc = 0.0
            for a, ti in zip(*self.shaper):
                p, v, ac = profile.pva_stretched(t - ti, self.duration)
                pos += a * p
                vel += a * v
                acc += a * ac
            q[j] = self.q_start[j] + pos
            qd[j], qdd[j] = vel, acc
        return q, qd, qdd

    def length_pva(self, t):
        if not self.extends:
            return self.length_start, 0.0, 0.0
        total = np.zeros(3)
        for a, ti in zip(*self.shaper):
            total += a * np.asarray(length_trapezoid_pva(float(t) - float(ti), self.length_start,
                                                         self.length_end, self.length_vmax,
                                                         self.length_amax))
        return float(total[0]), float(total[1]), float(total[2])
