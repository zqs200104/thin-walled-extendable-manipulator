"""Step 4 - Characterization of the deployment motion (Section 5.1).

Measurement (data/deployment_test/run_1 and run_2, SinglePoint1.ly): with
q1 = 0 deg and q2 = 90 deg, the boom is extended to 1 m at a commanded 30 mm/s,
held and retracted; the test was run twice. The script detects the extension,
hold and retraction from the 1 Hz low-pass trajectory and evaluates
  - travel, speed and return error (both runs);
  - the dominant Z frequency in 3 s windows against the deployed length (both
    runs), compared with the first natural frequency of the model at the same
    length (joints locked, linearized about the static equilibrium with gravity);
  - sinusoid fits of Z over the last 2.5 s of the extension and the 1.5 s after
    the stop (both runs);
  - the RMS amplitude of Z (above 1.5 Hz) in the same windows, measured and
    simulated;
  - the residual vibration in the hold window of run 2.

Simulation: the model with the joints locked at
(0, 90) deg is extended from 0.03 m to 1.00 m (spectral radius 0.9),
  (1) with the commanded 30 mm/s trapezoid, ramp time identified from the record;
  (2) with the recorded stroke of run 2 (low-pass 10 Hz), to test the effect of
      the measured speed variation;
and continued for 20 s after the stop.

Outputs in results/s04_deployment_test/:
  model_first_mode.csv     first natural frequency of the model versus length
  frequency_track.csv      per 3 s window: run, phase, deployed length, measured
                           peak frequency, model frequency, Z RMS above 1.5 Hz
Run time: about 3 min.
"""
import csv
import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, hilbert, sosfiltfilt

from stemarm import params as P, paths, simulate
from stemarm.commands import PointToPoint
from stemarm.mocap import read_nokov
from stemarm.runner import parser, run_jobs

RUNS = ("run_1", "run_2")


def data_path(run):
    return os.path.join(paths.DATA_DIR, "deployment_test", run, "SinglePoint1.ly")


DATA = data_path("run_2")          # hold window and recorded simulation stroke
OUT = paths.result_dir("s04_deployment_test")
Q_DEG = (0.0, 90.0)
L_START, L_END = 0.03, 1.00
COMMAND_SPEED = 0.030     # m/s, commanded from the host PC
HOLD_TIME = 20.0          # s, simulated after the stop
TRACK_WINDOW_S = 3.0      # s, windows of the frequency track
TRACK_STEP_S = 0.5        # s
TRACK_BAND_HZ = (2.5, 15.0)
HIGHPASS_HZ = 1.5
TRACK_LENGTHS_M = (0.45, 0.98)   # deployed lengths over which the track is compared with the model
STOP_FIT_S = (2.5, 1.5)          # s before / after the stop
MODEL_LENGTHS_M = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


# ----------------------------------------------------------------------------
# Measurement analysis
# ----------------------------------------------------------------------------
def lowpass(x, fs, cutoff):
    return sosfiltfilt(butter(4, cutoff, btype="lowpass", fs=fs, output="sos"), x, axis=0)


def highpass(x, fs, cutoff=HIGHPASS_HZ):
    return sosfiltfilt(butter(4, cutoff, btype="highpass", fs=fs, output="sos"), x, axis=0)


def find_runs(mask, max_gap):
    """Index ranges where mask holds; gaps up to max_gap samples are bridged."""
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        return []
    runs, start, previous = [], int(ids[0]), int(ids[0])
    for i in ids[1:]:
        if i - previous > max_gap:
            runs.append((start, previous))
            start = int(i)
        previous = int(i)
    return runs + [(start, previous)]


def motion_phases(t, trend, fs, threshold=5.0):
    """Extension and retraction: the two longest runs with speed above threshold [mm/s]."""
    velocity = np.gradient(trend, t, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    runs = [r for r in find_runs(speed > threshold, max(1, int(round(0.5 * fs)))) if t[r[1]] - t[r[0]] >= 1.0]
    runs = sorted(runs, key=lambda r: t[r[1]] - t[r[0]], reverse=True)[:2]
    extension, retraction = sorted(runs)
    return extension, retraction, velocity, speed


def strict_hold(t, xyz, threshold=1.0, smoothing=2.0):
    """Hold between the end of the outward motion and the start of the return (axial speed < 1 mm/s)."""
    dt = float(np.median(np.diff(t)))
    x = xyz[:, 0] - float(np.median(xyz[:max(5, int(round(1.0 / dt))), 0]))
    outward = (1.0 if abs(np.max(x)) >= abs(np.min(x)) else -1.0) * x
    n = max(3, int(round(smoothing / dt))) | 1
    axial_speed = np.gradient(np.convolve(outward, np.ones(n) / n, mode="same"), dt)
    out = max(find_runs(axial_speed > threshold, 1), key=lambda r: r[1] - r[0])
    back = np.flatnonzero((np.arange(len(t)) > out[1]) & (axial_speed < -threshold))
    return out[1], int(back[0])


def spectrum(values, dt):
    """Amplitude spectrum: mean removed, Hann window, coherent-gain correction."""
    centred = np.asarray(values, float) - np.mean(values)
    window = np.hanning(len(centred))
    amplitude = 2.0 * np.abs(np.fft.rfft(centred * window)) / (len(centred) * np.sum(window) / len(window))
    amplitude[0] *= 0.5
    if len(centred) % 2 == 0:
        amplitude[-1] *= 0.5
    return np.fft.rfftfreq(len(centred), dt), amplitude


def peak_frequency(x, fs, band=TRACK_BAND_HZ, pad=32):
    """Frequency of the largest peak of the Hann-windowed, zero-padded spectrum within band."""
    n = pad * len(x)
    amplitude = np.abs(np.fft.rfft((x - np.mean(x)) * np.hanning(len(x)), n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    keep = (f >= band[0]) & (f <= band[1])
    return float(f[keep][np.argmax(amplitude[keep])])


def sine_fit(t, x, fs):
    """Sinusoid with a linear trend, a cos(2 pi f t + phi) + c0 + c1 t; returns (f [Hz], a [mm])."""
    f0 = peak_frequency(x, fs, band=(2.5, 6.0))

    def model(t, a, f, phi, c0, c1):
        return a * np.cos(2.0 * np.pi * f * t + phi) + c0 + c1 * t

    best = None
    for phi in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        try:
            p, _ = curve_fit(model, t, x, p0=[np.sqrt(2.0) * np.std(x), f0, phi, 0.0, 0.0], maxfev=20000)
        except RuntimeError:
            continue
        err = float(np.sum((x - model(t, *p)) ** 2))
        if best is None or err < best[1]:
            best = (p, err)
    return float(abs(best[0][1])), float(abs(best[0][0]))


def stop_ramp(t, xyz, origin, unit, extension, fs, smooth_s=0.05):
    """Deceleration time at the end of the extension from the raw axial coordinate."""
    along = (xyz - origin) @ unit
    width = max(3, int(round(smooth_s * fs)) | 1)
    v = np.gradient(np.convolve(along, np.ones(width) / width, mode="same"), t)
    cruise = float(np.median(np.abs(v[extension[0]:extension[1] + 1])))
    stop = float(t[extension[1]])
    near = (t >= stop - 1.5) & (t <= stop + 0.5)
    fast = t[near][np.abs(v[near]) >= 0.9 * cruise]
    slow = t[near][np.abs(v[near]) <= 0.1 * cruise]
    if not len(fast) or not len(slow) or slow[0] <= fast[-1]:
        return 1.0 / fs
    return max(float(slow[0] - fast[-1]), 1.0 / fs)


def window_mean(t, xyz, lo, hi):
    keep = (t >= lo) & (t <= hi)
    return xyz[keep].mean(axis=0)


def analyse(path=DATA, verbose=True, label="run_2"):
    t, xyz = read_nokov(path)
    fs = 1.0 / np.median(np.diff(t))
    trend = lowpass(xyz, fs, 1.0)
    ext, ret, velocity, speed = motion_phases(t, trend, fs)
    hold = strict_hold(t, xyz)
    acceleration = np.linalg.norm(np.gradient(velocity, t, axis=0), axis=1)
    p0 = window_mean(t, xyz, max(t[0], t[ext[0]] - 3.0), t[ext[0]] - 1.0)
    lo, hi = t[ext[1]] + 1.0, t[ret[0]] - 1.0
    if hi - lo < 0.3:                      # short hold (run 1)
        lo, hi = t[ext[1]] + 0.5, t[ret[0]] - 0.2
    p1 = window_mean(t, xyz, lo, hi)
    pf = window_mean(t, xyz, min(t[-1] - 0.5, t[ret[1]] + 0.8), t[-1])
    disp = p1 - p0
    unit = disp / np.linalg.norm(disp)
    along = (trend - p0) @ unit
    axial_v = np.gradient(along, t)
    length = L_START + (L_END - L_START) * np.clip(along / along[ext[1]], 0.0, None)

    def central_speed(phase):
        ids = np.arange(phase[0], phase[1] + 1)
        lo, hi = np.quantile(along[ids], [0.10, 0.90])
        keep = ids[(along[ids] >= lo) & (along[ids] <= hi)]
        return float(np.mean(np.abs(axial_v[keep])))

    # residual vibration during the hold
    h = np.arange(hold[0], hold[1] + 1)
    residual = xyz[h] - np.mean(xyz[h], axis=0)
    f, a = spectrum(residual[:, 2], float(np.median(np.diff(t[h]))))
    band = (f >= 0.5) & (f <= min(10.0, fs / 2 - 0.1))
    i_peak = np.flatnonzero(band)[np.argmax(a[band])]

    # 2-5 Hz envelope of Z before the stop and during the hold
    zb = sosfiltfilt(butter(4, (2.0, 5.0), btype="bandpass", fs=fs, output="sos"), xyz[:, 2])
    envelope = np.abs(hilbert(zb))
    last5 = (t >= t[ext[1]] - 5.0) & (t <= t[ext[1]])
    in_hold = (t >= t[hold[0]]) & (t <= t[hold[1]])
    along_raw = (xyz - p0) @ unit
    acc10 = np.gradient(np.gradient(lowpass(along_raw, fs, 10.0), t), t)
    m_ext = (t >= t[ext[0]]) & (t <= t[ext[1]])

    r = {"t": t, "xyz": xyz, "trend": trend, "speed": speed, "acceleration": acceleration, "fs": fs,
         "ext": ext, "ret": ret, "hold": hold, "p0": p0, "unit": unit, "length": length,
         "residual_t": t[h] - t[h][0], "residual_z": residual[:, 2], "f": f[band], "a": a[band],
         "f_peak": float(f[i_peak]), "a_peak": float(a[i_peak]), "along_raw": along_raw,
         "ramp": stop_ramp(t, xyz, p0, unit, ext, fs),
         "window": (t[hold[0]] - t[ext[1]], t[hold[1]] - t[ext[1]])}
    if not verbose:
        return r
    print(f"  {label}: travel {np.linalg.norm(disp):.0f} mm, central speed extension {central_speed(ext):.1f} mm/s, "
          f"retraction {central_speed(ret):.1f} mm/s, return error {np.linalg.norm(pf - p0):.2f} mm, "
          f"retraction starts {t[ret[0]] - t[ext[1]]:.2f} s after the stop")
    print(f"  {label}: hold window {t[hold[0]] - t[ext[1]]:.2f}-{t[hold[1]] - t[ext[1]]:.2f} s after the stop "
          f"({t[hold[1]] - t[hold[0]]:.2f} s): Z RMS {np.sqrt(np.mean(residual[:, 2]**2)):.2f} mm, "
          f"peak-to-peak {np.ptp(residual[:, 2]):.2f} mm, spectral peak {r['f_peak']:.2f} Hz "
          f"(bins {1.0 / (t[hold[1]] - t[hold[0]]):.2f} Hz apart)")
    print(f"  {label}: 2-5 Hz envelope median: last 5 s of extension {np.median(envelope[last5]):.2f} mm, "
          f"hold {np.median(envelope[in_hold]):.2f} mm; axial acceleration after 10 Hz low-pass: "
          f"max {np.max(np.abs(acc10[m_ext])):.0f} mm/s^2; measured ramp time {r['ramp']:.3f} s")
    return r


# ----------------------------------------------------------------------------
# Frequency versus deployed length
# ----------------------------------------------------------------------------
def model_first_mode(force=False):
    """First natural frequency of the model at q = (0, 90) deg versus the deployed length (cached CSV)."""
    path = os.path.join(OUT, "model_first_mode.csv")
    if not os.path.isfile(path) or force:
        from stemarm.modal import damped_modes
        rows = []
        for length in MODEL_LENGTHS_M:
            f_n, f_d, zeta = damped_modes(np.deg2rad(Q_DEG), length, gravity=True, n_modes=1)[0]
            rows.append([length, f_n, f_d, zeta])
            print(f"  model first mode at l = {length:.2f} m: {f_d:.3f} Hz (damped), zeta {zeta:.4f}")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["length_m", "natural_frequency_hz", "damped_frequency_hz", "damping_ratio"])
            w.writerows(rows)
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 2]


def model_frequency(length, table):
    """Model frequency at length, interpolated on log-log axes."""
    lengths, freqs = table
    return np.exp(np.interp(np.log(length), np.log(lengths), np.log(freqs)))


def frequency_track(m):
    """Peak frequency and RMS of Z (above 1.5 Hz) in sliding windows during extension and retraction."""
    t, fs = m["t"], m["fs"]
    z = highpass(m["xyz"][:, 2], fs)
    rows = []
    for phase, (i0, i1) in (("extension", m["ext"]), ("retraction", m["ret"])):
        for c in np.arange(t[i0] + TRACK_WINDOW_S / 2, t[i1] - TRACK_WINDOW_S / 2, TRACK_STEP_S):
            w = (t >= c - TRACK_WINDOW_S / 2) & (t < c + TRACK_WINDOW_S / 2)
            rows.append((phase, float(np.mean(m["length"][w])), peak_frequency(z[w], fs),
                         float(np.sqrt(np.mean(z[w] ** 2)))))
    return rows


def stop_fits(m):
    """Sinusoid fits of Z over the last 2.5 s of the extension and the 1.5 s after the stop."""
    t, fs = m["t"], m["fs"]
    z = highpass(m["xyz"][:, 2], fs)
    stop = t[m["ext"][1]]
    out = []
    for lo, hi in ((stop - STOP_FIT_S[0], stop), (stop, stop + STOP_FIT_S[1])):
        w = (t >= lo) & (t < hi)
        out.append((float(np.mean(m["length"][w])), *sine_fit(t[w] - t[w][0], z[w], fs)))
    return out


def simulated_track_rms(name):
    """RMS of the simulated tip Z (above 1.5 Hz) in the windows of the track, within TRACK_LENGTHS_M."""
    r = simulate.load(paths.result_dir("s04_deployment_test", name))
    t, end = r["t"], r["meta"]["command_end_s"]
    fs = 1.0 / float(np.median(np.diff(t)))
    z = highpass(1e3 * r["tip"][:, 2], fs)
    length = L_START + (L_END - L_START) * np.clip(t / end, 0.0, 1.0)
    out = []
    for c in np.arange(TRACK_WINDOW_S / 2, end - TRACK_WINDOW_S / 2, TRACK_STEP_S):
        w = (t >= c - TRACK_WINDOW_S / 2) & (t < c + TRACK_WINDOW_S / 2)
        if TRACK_LENGTHS_M[0] <= np.mean(length[w]) <= TRACK_LENGTHS_M[1]:
            out.append(float(np.sqrt(np.mean(z[w] ** 2))))
    return max(out)


def length_analysis(records, table):
    """Frequency track of both runs against the model, the stop fits and the amplitudes."""
    rows, ratios = [], []
    print(f"  dominant Z frequency in {TRACK_WINDOW_S:.0f} s windows, l in {TRACK_LENGTHS_M} m, measured / model:")
    for run, m in records.items():
        track = frequency_track(m)
        for phase, length, f, rms in track:
            f_model = float(model_frequency(length, table))
            rows.append([run, phase, length, f, f_model, rms])
        for phase in ("extension", "retraction"):
            sel = [r for r in rows if r[0] == run and r[1] == phase and TRACK_LENGTHS_M[0] <= r[2] <= TRACK_LENGTHS_M[1]]
            q = np.array([r[3] / r[4] for r in sel])
            ratios.extend(q)
            short = [r[5] for r in sel if r[2] < 0.55]
            long_ = [r[5] for r in sel if r[2] > 0.90]
            print(f"    {run} {phase:10s}: n {len(q)}, median {np.median(q):.3f}, 10-90 % {np.percentile(q, 10):.3f}-"
                  f"{np.percentile(q, 90):.3f}; Z RMS median {np.median(short):.3f} mm (l < 0.55 m), "
                  f"{np.median(long_):.3f} mm (l > 0.90 m)")
    q = np.array(ratios)
    print(f"    all: n {len(q)}, median {np.median(q):.3f}, 10-90 % {np.percentile(q, 10):.3f}-{np.percentile(q, 90):.3f}")
    with open(os.path.join(OUT, "frequency_track.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "phase", "length_m", "peak_frequency_hz", "model_frequency_hz", "z_rms_mm"])
        w.writerows(rows)
    print("  sinusoid fits of Z before / after the stop:")
    for run, m in records.items():
        (l0, f0, a0), (l1, f1, a1) = stop_fits(m)
        print(f"    {run}: last {STOP_FIT_S[0]} s of extension {f0:.3f} Hz, {a0:.2f} mm (l {l0:.3f} m, model "
              f"{float(model_frequency(l0, table)):.3f} Hz); first {STOP_FIT_S[1]} s after the stop {f1:.3f} Hz, "
              f"{a1:.2f} mm (model {float(model_frequency(l1, table)):.3f} Hz)")
    for name in ("commanded", "recorded_stroke"):
        if simulate.is_done(paths.result_dir("s04_deployment_test", name)):
            print(f"  simulation, {name}: largest Z RMS (above {HIGHPASS_HZ} Hz) in the track windows "
                  f"{simulated_track_rms(name):.4f} mm")
    return rows


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
class RecordedStroke:
    """Deployed length following the recorded stroke (low-pass 10 Hz), joints locked."""

    def __init__(self, t, along_mm, extension, fs):
        ids = np.arange(extension[0], extension[1] + 1)
        self.tau = t[ids] - t[ids[0]]
        stroke = lowpass(along_mm[ids], fs, 10.0)
        stroke = stroke - stroke[0]
        self.stroke = stroke * (1000.0 * (L_END - L_START) / float(stroke[-1])) / 1000.0
        self.end_time = float(self.tau[-1])
        self.q_start = np.deg2rad(Q_DEG)

    def joint_pva(self, t):
        return self.q_start, np.zeros(2), np.zeros(2)

    def length_pva(self, t):
        return L_START + float(np.interp(float(t), self.tau, self.stroke, left=0.0, right=self.stroke[-1])), 0.0, 0.0


def simulate_case(name, force=False, render: bool = False):
    folder = paths.result_dir("s04_deployment_test", name)
    if simulate.is_done(folder) and not (force or render):
        print(f"  {name}: stored result used")
        return
    m = analyse_cached()
    if name == "commanded":
        command = PointToPoint(Q_DEG, Q_DEG, length=L_START, length_end=L_END, length_vmax=COMMAND_SPEED,
                               length_amax=COMMAND_SPEED / m["ramp"])
    else:
        command = RecordedStroke(m["t"], m["along_raw"], m["ext"], m["fs"])
    result = simulate.run(command, command.q_start, L_START, command.end_time + HOLD_TIME, gravity=True,
                          joint_drive="constraint", variable_length=True, spectral_radius=0.9, verbose=0,
                          render=render, render_pause=False)
    simulate.save(result, folder, {"command": name, "command_end_s": command.end_time, "ramp_s": m["ramp"]})


_CACHE = {}


def analyse_cached():
    if "m" not in _CACHE:
        _CACHE["m"] = analyse(verbose=False)
    return _CACHE["m"]


def evaluate_simulation(m):
    rms = {}
    for name in ("commanded", "recorded_stroke"):
        r = simulate.load(paths.result_dir("s04_deployment_test", name))
        tau = r["t"] - r["meta"]["command_end_s"]
        z = 1e3 * r["tip"][:, 2]
        out = []
        for lo, hi in (m["window"], (0.0, HOLD_TIME)):
            keep = (tau >= lo) & (tau <= hi)
            zz = z[keep] - z[keep].mean()
            window = np.hanning(len(zz))
            amp = 2 * np.abs(np.fft.rfft(zz * window)) / window.sum()
            f = np.fft.rfftfreq(len(zz), float(np.median(np.diff(tau[keep]))))
            band = (f >= 0.5) & (f <= 10.0)
            out.append((float(np.sqrt(np.mean(zz**2))), float(f[np.flatnonzero(band)[np.argmax(amp[band])]])))
        rms[name] = out
        print(f"  simulation, {name:15s}: hold window RMS {out[0][0]:.4f} mm, peak {out[0][1]:.2f} Hz; "
              f"0-20 s RMS {out[1][0]:.4f} mm, peak {out[1][1]:.2f} Hz")
    change = 100 * (rms["recorded_stroke"][0][0] / rms["commanded"][0][0] - 1)
    print(f"  recorded stroke instead of the trapezoid changes the window RMS by {change:+.1f} %")
    r = simulate.load(paths.result_dir("s04_deployment_test", "commanded"))
    tau = r["t"] - r["meta"]["command_end_s"]
    keep = (tau >= 0.0) & (tau <= 5.0)
    t, z = tau[keep], 1e3 * r["tip"][keep, 2]
    z = z - z.mean()
    f0 = peak_frequency(z, 1.0 / float(np.median(np.diff(t))), band=(2.5, 6.0))

    def decay(t, a, s, f, phi, c):
        return a * np.exp(-s * t) * np.cos(2.0 * np.pi * f * t + phi) + c

    best = None
    for phi in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        try:
            p, _ = curve_fit(decay, t, z, p0=[np.sqrt(2.0) * np.std(z), 0.3, f0, phi, 0.0], maxfev=20000)
        except RuntimeError:
            continue
        err = float(np.sum((z - decay(t, *p)) ** 2))
        if best is None or err < best[1]:
            best = (p, err)
    print(f"  simulated residual after the stop rings at {abs(best[0][2]):.3f} Hz (decaying sinusoid over 0-5 s)")


def main():
    args = parser(__doc__).parse_args()
    records = {run: analyse(data_path(run), label=run) for run in RUNS}
    m = records["run_2"]
    _CACHE["m"] = m
    run_jobs(simulate_case, [dict(name=n, force=args.force, render=args.render) for n in ("commanded", "recorded_stroke")],
             1 if args.render else args.workers)
    table = model_first_mode(force=args.force)
    length_analysis(records, table)
    if all(simulate.is_done(paths.result_dir("s04_deployment_test", n)) for n in ("commanded", "recorded_stroke")):
        evaluate_simulation(m)


if __name__ == "__main__":
    main()
