"""Input shapers (Section 3.1) and trapezoidal reference profiles.

A shaper is a pair (A, T) of impulse amplitudes and times; a shaped command is
sum_i A_i * r(t - T_i).
"""
import numpy as np


# ----------------------------------------------------------------------------
# ZVD and ZVD2 shapers, Eqs. (35)-(37)
# ----------------------------------------------------------------------------
def zvd(fn, zeta):
    """ZVD shaper for natural frequency fn [Hz] and damping ratio zeta."""
    wd = 2.0 * np.pi * fn * np.sqrt(max(1.0 - zeta**2, 1e-12))
    half_period = np.pi / wd
    k = np.exp(-zeta * np.pi / np.sqrt(max(1.0 - zeta**2, 1e-12)))
    amplitudes = np.array([1.0, 2.0 * k, k**2]) / (1.0 + k)**2
    times = np.array([0.0, half_period, 2.0 * half_period])
    return amplitudes, times


def convolve(shaper_1, shaper_2):
    """Convolution of two shapers, impulses sorted by time."""
    a = np.array([a1 * a2 for a1 in shaper_1[0] for a2 in shaper_2[0]], dtype=float)
    t = np.array([t1 + t2 for t1 in shaper_1[1] for t2 in shaper_2[1]], dtype=float)
    order = np.argsort(t)
    return a[order], t[order]


def merge_equal_times(shaper, tol=1e-12):
    """Merge impulses at the same time and normalise the amplitudes to sum 1."""
    a, t = shaper
    a_new, t_new = [a[0]], [t[0]]
    for ak, tk in zip(a[1:], t[1:]):
        if abs(tk - t_new[-1]) <= tol:
            a_new[-1] += ak
        else:
            a_new.append(ak)
            t_new.append(tk)
    a_new = np.array(a_new, dtype=float)
    return a_new / np.sum(a_new), np.array(t_new, dtype=float)


def zvd2(fn_1, fn_2, zeta_1=0.01, zeta_2=0.01):
    """Two-frequency shaper ZVD2 = ZVD(fn_1) * ZVD(fn_2), Eq. (37)."""
    return merge_equal_times(convolve(zvd(fn_1, zeta_1), zvd(fn_2, zeta_2)))


def make_shaper(kind, frequencies=(), zeta=0.01):
    """Shaper by name: 'none', 'zvd' (first frequency) or 'zvd2' (first two)."""
    kind = kind.lower()
    if kind == "none":
        return np.array([1.0]), np.array([0.0])
    if kind == "zvd":
        return zvd(frequencies[0], zeta)
    if kind == "zvd2":
        return zvd2(frequencies[0], frequencies[1], zeta, zeta)
    raise ValueError(f"unknown shaper {kind!r}")


def duration(shaper):
    """Shaper duration t_N."""
    return float(np.max(shaper[1]))


# ----------------------------------------------------------------------------
# Rest-to-rest trapezoidal velocity profile
# ----------------------------------------------------------------------------
class Trapezoid:
    """Rest-to-rest move of `distance` with limits vmax and amax.

    If the distance is too short to reach vmax the profile is triangular.
    `pva(t)` returns position, velocity and acceleration.
    """

    def __init__(self, distance, vmax, amax):
        self.sign = 1.0 if distance >= 0 else -1.0
        self.distance = abs(distance)
        self.a = amax
        if self.distance < 1e-15:
            self.t_acc = self.t_flat = self.v_peak = self.d_acc = 0.0
        else:
            t_acc = vmax / amax
            d_acc = 0.5 * amax * t_acc**2
            if self.distance >= 2.0 * d_acc:
                self.t_acc, self.v_peak = t_acc, vmax
                self.t_flat = (self.distance - 2.0 * d_acc) / vmax
                self.d_acc = d_acc
            else:
                self.t_acc = np.sqrt(self.distance / amax)
                self.v_peak = amax * self.t_acc
                self.t_flat = 0.0
                self.d_acc = 0.5 * amax * self.t_acc**2
        self.total_time = 2.0 * self.t_acc + self.t_flat

    def pva(self, t):
        if t <= 0.0:
            return 0.0, 0.0, 0.0
        if t >= self.total_time:
            return self.sign * self.distance, 0.0, 0.0
        if t < self.t_acc:
            pos, vel, acc = 0.5 * self.a * t**2, self.a * t, self.a
        elif t < self.t_acc + self.t_flat:
            tau = t - self.t_acc
            pos, vel, acc = self.d_acc + self.v_peak * tau, self.v_peak, 0.0
        else:
            tau = t - (self.t_acc + self.t_flat)
            pos = self.d_acc + self.v_peak * self.t_flat + self.v_peak * tau - 0.5 * self.a * tau**2
            vel, acc = self.v_peak - self.a * tau, -self.a
        return self.sign * pos, self.sign * vel, self.sign * acc

    def pva_stretched(self, t, duration_common):
        """The same move uniformly stretched in time to `duration_common`."""
        if self.total_time <= 1e-12 or duration_common <= 1e-12:
            return self.sign * self.distance, 0.0, 0.0
        if t <= 0.0:
            return 0.0, 0.0, 0.0
        if t >= duration_common:
            return self.sign * self.distance, 0.0, 0.0
        scale = self.total_time / duration_common
        pos, vel, acc = self.pva(float(t) * scale)
        return pos, vel * scale, acc * scale**2


def length_trapezoid_pva(t, l_start, l_end, vmax, amax):
    """Deployed length of a one-way trapezoidal extension (used in Section 4.5)."""
    distance = float(l_end - l_start)
    v_peak = min(vmax, np.sqrt(distance * amax))
    t_acc = v_peak / amax
    d_acc = 0.5 * amax * t_acc * t_acc
    t_flat = max(0.0, distance - 2.0 * d_acc) / v_peak
    t_end = 2.0 * t_acc + t_flat
    t = float(t)
    if t <= 0.0:
        s, v, a = 0.0, 0.0, 0.0
    elif t < t_acc:
        s, v, a = 0.5 * amax * t * t, amax * t, amax
    elif t < t_acc + t_flat:
        tau = t - t_acc
        s, v, a = d_acc + v_peak * tau, v_peak, 0.0
    elif t < t_end:
        tau = t - (t_acc + t_flat)
        s, v, a = d_acc + v_peak * t_flat + v_peak * tau - 0.5 * amax * tau * tau, v_peak - amax * tau, -amax
    else:
        s, v, a = distance, 0.0, 0.0
    return min(l_start + s, l_end), v, a


def length_trapezoid_time(l_start, l_end, vmax, amax):
    distance = float(l_end - l_start)
    v_peak = min(vmax, np.sqrt(distance * amax))
    t_acc = v_peak / amax
    d_acc = 0.5 * amax * t_acc * t_acc
    return 2.0 * t_acc + max(0.0, distance - 2.0 * d_acc) / v_peak


# ----------------------------------------------------------------------------
# Path-progress shaping of trajectory commands, Eqs. (39)-(45)
# ----------------------------------------------------------------------------
def uniform_time_grid(end, dt):
    """Samples 0, dt, 2 dt, ... ending exactly at `end`.

    If `end` lies within 1e-3 dt of the last multiple of dt, that sample is
    moved to `end`; a second knot a rounding error away would make the second
    derivative of the Hermite spline explode.
    """
    times = np.arange(0.0, float(end), float(dt))
    if len(times) == 0 or times[-1] < end - 1.0e-3 * dt:
        return np.append(times, float(end))
    times[-1] = float(end)
    return times


def shape_path_progress(times, progress, shaper, dt, t_acc=0.0):
    """Compress s(t) by k = (T - t_acc)/T, then convolve with the shaper.

    Returns the shaped time grid, S(t) on that grid and k. The shaped motion
    lasts k T + t_N = T - t_acc + t_N, so t_acc = t_N ends it together with the
    unshaped one (zero time delay).
    """
    times = np.asarray(times, dtype=float)
    progress = np.asarray(progress, dtype=float)
    total = float(times[-1])
    if not 0.0 <= float(t_acc) < total:
        raise ValueError(f"t_acc = {t_acc} s must lie in [0, T = {total} s)")
    k = (total - float(t_acc)) / total
    amplitudes, delays = shaper
    shaped_times = uniform_time_grid(k * total + float(np.max(delays)), dt)
    shaped = np.zeros(len(shaped_times))
    for amplitude, delay in zip(amplitudes, delays):
        shaped += float(amplitude) * np.interp((shaped_times - float(delay)) / k, times, progress,
                                               left=0.0, right=1.0)
    shaped = np.maximum.accumulate(np.clip(shaped, 0.0, 1.0))
    shaped[0], shaped[-1] = 0.0, 1.0
    return shaped_times, shaped, k
