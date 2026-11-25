"""Residual-vibration measures of Section 3.3 and common signal metrics."""
import numpy as np

# np.trapz was renamed np.trapezoid in NumPy 2.0
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def one_sided_fft(x, dt, window=True, remove_mean=True):
    """Amplitude spectrum with a Hann window corrected by its coherent gain."""
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    if n < 4 or dt <= 0:
        return np.array([]), np.array([])
    if remove_mean:
        x = x - np.mean(x)
    gain = 1.0
    if window:
        w = np.hanning(n)
        gain = np.sum(w) / n
        x = x * w
    spectrum = np.fft.rfft(x)
    amplitude = 2.0 * np.abs(spectrum) / (n * gain)
    amplitude[0] = np.abs(spectrum[0]) / (n * gain)
    return np.fft.rfftfreq(n, d=dt), amplitude


def residual_spectrum(tau, value, duration):
    """Spectrum of value(tau) over tau in [0, duration], resampled to a uniform grid."""
    tau, value = _finite_sorted(tau, value)
    keep = (tau >= 0.0) & (tau <= float(duration))
    tau, value = tau[keep], value[keep]
    if len(tau) < 8:
        return np.array([]), np.array([])
    dt = float(np.median(np.diff(tau)))
    grid = np.arange(tau[0], tau[-1] + 0.5 * dt, dt)
    uniform = np.interp(grid, tau, value)
    return one_sided_fft(uniform - np.mean(uniform), dt, window=True, remove_mean=False)


def top_peaks(frequency, amplitude, count=3, f_min=0.1):
    """The `count` largest spectral bins above f_min, largest first."""
    frequency, amplitude = np.asarray(frequency, float), np.asarray(amplitude, float)
    band = np.isfinite(frequency) & np.isfinite(amplitude) & (frequency >= f_min)
    f, a = frequency[band], amplitude[band]
    if len(a) < 3 or not np.any(a > 0.0):
        return []
    order = np.argsort(a)[::-1][:count]
    return [(float(f[i]), float(a[i])) for i in order]


def _finite_sorted(tau, value):
    tau, value = np.asarray(tau, float), np.asarray(value, float)
    good = np.isfinite(tau) & np.isfinite(value)
    order = np.argsort(tau[good])
    return tau[good][order], value[good][order]


def clip_window(tau, value, lo, hi):
    """Restrict a curve to [lo, hi]; the end points are interpolated."""
    tau, value = _finite_sorted(tau, value)
    if len(tau) < 2:
        return np.array([]), np.array([])
    lo, hi = max(float(lo), tau[0]), min(float(hi), tau[-1])
    if hi <= lo:
        return np.array([]), np.array([])
    inside = (tau > lo) & (tau < hi)
    return (np.concatenate(([lo], tau[inside], [hi])),
            np.concatenate(([np.interp(lo, tau, value)], value[inside], [np.interp(hi, tau, value)])))


def window_rms(tau, residual, lo, hi):
    """Time-weighted RMS (trapezoidal rule), peak-to-peak and peak over [lo, hi].

    With residual = z - z_eq this is J_RMS of Eq. (48).
    """
    t, v = clip_window(tau, residual, lo, hi)
    if len(t) < 2 or t[-1] <= t[0]:
        return None
    return {"rms": float(np.sqrt(_trapezoid(v**2, t) / (t[-1] - t[0]))),
            "p2p": float(np.max(v) - np.min(v)), "max_abs": float(np.max(np.abs(v))),
            "tau_min": float(t[0]), "tau_max": float(t[-1])}


def median_baseline(tau, value, lo, hi):
    """Steady-state value z_eq: median over the baseline window W_b = [lo, hi]."""
    tau, value = _finite_sorted(tau, value)
    keep = (tau >= lo) & (tau <= hi)
    if np.count_nonzero(keep) < 3:
        raise ValueError("fewer than three samples in the baseline window")
    return float(np.median(value[keep]))


def tracking_metric(t_sim, y_sim, t_exp, y_exp, lo, hi):
    """RMSE, MAE and R^2 of a simulated signal against a measured one on [lo, hi].

    The measurement is interpolated onto the simulation time stamps.
    """
    t_sim, y_sim = _finite_sorted(t_sim, y_sim)
    t_exp, y_exp = _finite_sorted(t_exp, y_exp)
    lo = max(float(lo), t_sim[0], t_exp[0])
    hi = min(float(hi), t_sim[-1], t_exp[-1])
    keep = (t_sim >= lo) & (t_sim <= hi)
    sim = y_sim[keep]
    exp = np.interp(t_sim[keep], t_exp, y_exp)
    err = sim - exp
    ss_tot = float(np.sum((exp - np.mean(exp))**2))
    rmse = float(np.sqrt(np.mean(err**2)))
    span = float(np.ptp(exp))
    return {"rmse": rmse, "mae": float(np.mean(np.abs(err))), "max_abs": float(np.max(np.abs(err))),
            "nrmse_percent": 100.0 * rmse / span if span > 1e-12 else np.nan,
            "r2": 1.0 - float(np.sum(err**2)) / ss_tot if ss_tot > 1e-12 else np.nan,
            "tau_min": lo, "tau_max": hi, "n": int(len(sim))}


def zero_crossing_frequency(t, x, duration):
    """Frequency from the upward zero crossings of x over the first `duration` s.

    The mean and a linear trend are removed first.
    """
    t, x = np.asarray(t, float), np.asarray(x, float)
    keep = t <= t[0] + duration
    t, x = t[keep], x[keep] - np.mean(x[keep])
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    crossings = t[:-1][(x[:-1] < 0.0) & (x[1:] >= 0.0)]
    return (len(crossings) - 1) / (crossings[-1] - crossings[0])

