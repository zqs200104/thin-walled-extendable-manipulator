"""Step 8 - Slews of a 5 m boom under microgravity (Section 4.4).

The slews of Cases A and B are repeated with L = 5 m and no gravity on the
links, the boom and the gripper; root stiffness and damping are unchanged.
The shapers are designed from the unshaped model response alone: ZVD at
0.45 Hz and ZVD2 at 0.45 + 0.50 Hz (zeta = 0.01). The time step is 0.25 ms.

Also printed (Section 4.4): decaying-sinusoid fits of the unshaped Z response
(frequency and damping ratio in tau windows [0, 5], [5, 10], [10, 20] s), the
residual amplitude that ZVD and ZVD2 leave for a single damped mode at
0.40-0.55 Hz, and the duration of the unshaped commands.

Results: results/s08_microgravity_5m/case_<A|B>_<shaper>/
Run time: about 35 min per case on one core (six cases in parallel).
"""
import csv
import os

import numpy as np
from scipy.optimize import curve_fit

from stemarm import params as P, paths, shaping, simulate
from stemarm import evaluation as ev
from stemarm.commands import PointToPoint
from stemarm.runner import parser, run_jobs

LENGTH = 5.0                       # m
STEP_SIZE = 0.25e-3                # s
SHAPER_FREQUENCIES = (0.45, 0.50)  # Hz, the two highest bins of the unshaped Z spectrum
SHAPER_ZETA = 0.01
POST_COMMAND_TIME = 20.0           # s
SHAPERS = ("none", "zvd", "zvd2")
OUT = paths.result_dir("s08_microgravity_5m")


def case_folder(case, shaper_name):
    return paths.result_dir("s08_microgravity_5m", f"case_{case}_{shaper_name}")


def simulate_case(case, shaper_name, force=False, render: bool = False):
    folder = case_folder(case, shaper_name)
    if simulate.is_done(folder) and not (force or render):
        print(f"  case {case} {shaper_name}: stored result used")
        return
    shaper = shaping.make_shaper(shaper_name, SHAPER_FREQUENCIES, SHAPER_ZETA)
    spec = P.CASES[case]
    command = PointToPoint(spec["q_start_deg"], spec["q_end_deg"], shaper=shaper, length=LENGTH)
    print(f"  case {case} {shaper_name}: command ends at {command.end_time:.4f} s "
          f"(shaper duration {shaping.duration(shaper):.2f} s)")
    result = simulate.run(command, command.q_start, LENGTH, command.end_time + POST_COMMAND_TIME,
                          gravity=False, joint_drive="constraint", step_size=STEP_SIZE,
                          modified_newton=False, verbose=0,
                          render=render, render_pause=False)
    simulate.save(result, folder, {
        "case": case, "shaper": shaper_name, "frequencies_hz": list(SHAPER_FREQUENCIES),
        "zeta": SHAPER_ZETA, "command_end_s": command.end_time, "length_m": LENGTH,
        "gravity": False, "step_size_s": STEP_SIZE})


def command_end_sample(result):
    """tau = 0 at the last stored sample at or before the command end."""
    t = result["t"]
    return float(t[t <= result["meta"]["command_end_s"] + 1e-9][-1])


def tail(result):
    """Tip-node Z after the command end [mm] and its residual about the last 2 s median."""
    t_end = command_end_sample(result)
    keep = result["t"] >= t_end
    tau, z = result["t"][keep] - t_end, 1e3 * result["tip"][keep]
    residual = z - np.median(z[tau >= tau[-1] - 2.0], axis=0)
    return tau, z, residual


def evaluate():
    """J_RMS over tau in [0, 20] s with z_eq the mean over that window (Section 4.4)."""
    rows = []
    for case in ("A", "B"):
        reference = None
        for shaper in SHAPERS:
            result = simulate.load(case_folder(case, shaper))
            tau, z, _ = tail(result)
            keep = tau <= POST_COMMAND_TIME
            j = [ev.window_rms(tau, z[:, i] - np.mean(z[keep, i]), 0.0, POST_COMMAND_TIME)["rms"]
                 for i in range(3)]
            reference = reference or j[2]
            theta_max = np.rad2deg(np.max(np.abs(result["theta_root"])))
            rows.append([case, shaper, j[0], j[1], j[2], 100.0 * (1.0 - j[2] / reference), theta_max])
            print(f"  case {case} {shaper:5s}: J_RMS Z = {j[2]:8.2f} mm (X {j[0]:.2f}, Y {j[1]:.2f}), "
                  f"eta = {rows[-1][5]:6.2f} %, max root rotation {theta_max:.1f} deg")
    with open(os.path.join(OUT, "residual_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "shaper", "J_rms_x_mm", "J_rms_y_mm", "J_rms_z_mm", "eta_rms_z_percent",
                    "max_root_rotation_deg"])
        w.writerows(rows)


def decay_fit(tau, x, lo, hi):
    """Decaying sinusoid a exp(-s t) cos(2 pi f t + phi) + c fitted over [lo, hi]; returns (f [Hz], zeta)."""
    k = (tau >= lo) & (tau <= hi)
    t, y = tau[k] - lo, x[k] - np.mean(x[k])
    n = 16 * len(y)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), n))
    freq = np.fft.rfftfreq(n, float(np.median(np.diff(t))))
    band = (freq > 0.1) & (freq < 2.0)
    f0 = freq[band][np.argmax(spec[band])]

    def model(t, a, s, f, phi, c):
        return a * np.exp(-s * t) * np.cos(2.0 * np.pi * f * t + phi) + c

    best = None
    for phi in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        try:
            p, _ = curve_fit(model, t, y, p0=[np.sqrt(2.0) * np.std(y), 0.01, f0, phi, 0.0], maxfev=20000)
        except RuntimeError:
            continue
        err = float(np.sum((y - model(t, *p)) ** 2))
        if best is None or err < best[1]:
            best = (p, err)
    _, s, f, _, _ = best[0]
    return abs(f), s / (2.0 * np.pi * abs(f))


def single_mode_residual(shaper, f, zeta=SHAPER_ZETA):
    """Residual amplitude left by the shaper for a single damped mode at f, relative to an unshaped impulse."""
    a, t = shaper
    w = 2.0 * np.pi * f
    wd = w * np.sqrt(1.0 - zeta ** 2)
    c = np.sum(a * np.exp(zeta * w * t) * np.cos(wd * t))
    s = np.sum(a * np.exp(zeta * w * t) * np.sin(wd * t))
    return float(np.exp(-zeta * w * t[-1]) * np.hypot(c, s))


def frequency_check():
    """Unshaped frequency and damping, shaper robustness for a single mode, command durations (Section 4.4)."""
    print("  unshaped Z, decaying-sinusoid fits over tau [0, 5], [5, 10], [10, 20] s (frequency / damping ratio):")
    for case in ("A", "B"):
        tau, z, residual = tail(simulate.load(case_folder(case, "none")))
        fits = [decay_fit(tau, z[:, 2], lo, hi) for lo, hi in ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0))]
        print(f"    case {case}: " + ", ".join(f"{f:.4f} Hz / {zeta:.4f}" for f, zeta in fits))
        f, amplitude = ev.one_sided_fft(residual[:, 2], float(np.median(np.diff(tau))))
        keep = (f > 0.0) & (f <= 1.0)
        peaks = np.argsort(amplitude[keep])[::-1][:2]
        print(f"    case {case}: two largest unshaped bins: "
              + ", ".join(f"{value:.2f} Hz" for value in f[keep][peaks]))
    print(f"  residual amplitude left for a single damped mode (zeta = {SHAPER_ZETA}):")
    for name in ("zvd", "zvd2"):
        shaper = shaping.make_shaper(name, SHAPER_FREQUENCIES, SHAPER_ZETA)
        print(f"    {name:5s}: " + ", ".join(f"{f:.2f} Hz {100.0 * single_mode_residual(shaper, f):.2f} %"
                                          for f in (0.40, 0.45, 0.50, 0.55)))
    for case in ("A", "B"):
        spec = P.CASES[case]
        command = PointToPoint(spec["q_start_deg"], spec["q_end_deg"], length=LENGTH)
        print(f"    case {case}: unshaped command {command.end_time:.2f} s")


def main():
    args = parser(__doc__).parse_args()
    jobs = [dict(case=c, shaper_name=s, force=args.force, render=args.render) for c in ("A", "B") for s in SHAPERS]
    run_jobs(simulate_case, jobs, 1 if args.render else args.workers)
    evaluate()
    frequency_check()


if __name__ == "__main__":
    main()
