"""Step 6 - Simulation-experiment comparison of the unshaped slews (Section 5.2).

Needs the results of s05_ground_slews.py and the unshaped measurements in
data/shaping_tests (first trial of each case, 2026-11-13). For these slews
the marker point P of the simulation is shifted along the boom axis by
P.MARKER_AXIAL_SHIFT_SLEW_MM; the dynamics are not affected.

Outputs in results/s06_model_validation/:
  Table 4  table3_residual_z.csv                              RMS, peak and dominant frequency
  motion_metrics.csv   RMSE / R^2 of the joint angles and tip coordinates quoted in the text
  residual_window_fits.csv  decaying sinusoids fitted to tau windows [0,2], [2,5], [5,10], [10,20] s
                       of the Z residual (simulation and the three unshaped trials): amplitude at
                       the window start, frequency, damping ratio
  command_excitation.csv    free amplitude that the commanded joint-2 angle leaves on a single mode
                       (zeta = 0.013) versus the mode frequency, 2.8-4.0 Hz
"""
import argparse
import csv
import os

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

from stemarm import params as P, paths, simulate, shaping
from stemarm import evaluation as ev
from stemarm.commands import PointToPoint
from stemarm.mocap import load_trial
from stemarm.runner import add_render_argument

OUT = paths.result_dir("s06_model_validation")
DATA = os.path.join(paths.DATA_DIR, "shaping_tests")
WINDOWS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0))   # s, windows of the decay fits
ZETA_MEASURED = 0.013          # free-decay tests, Section 4.3
R_TIP_MM = 1200.0              # approximate distance of P from the joint-2 axis (only scales the amplitude)


def shifted_marker(sim):
    """Marker point P moved by MARKER_AXIAL_SHIFT_SLEW_MM along the boom axis (tip-node x) [m]."""
    shift = np.array([1e-3 * P.MARKER_AXIAL_SHIFT_SLEW_MM, 0.0, 0.0])
    return sim["marker"] + np.einsum("nij,j->ni", sim["tip_rot"], shift)


def detected_command_end(t, command):
    """Last stored sample at which the commanded joint velocity is not zero."""
    moving = [np.any(np.abs(command.joint_pva(tk)[1]) > 1e-10) for tk in t]
    return float(t[np.flatnonzero(moving)[-1]])


def load_case(case):
    """Unshaped simulation and first measured trial of one case, aligned in tau."""
    sim = simulate.load(paths.result_dir("s05_ground_slews", f"case_{case}_none"))
    spec = P.CASES[case]
    command = PointToPoint(spec["q_start_deg"], spec["q_end_deg"], length=1.0)
    t_end = float(sim["meta"]["command_end_s"])
    exp = load_trial(os.path.join(DATA, f"case_{case}", "none", "1"))
    enc = exp["encoder"]
    rec = {
        "case": case,
        # motion record: tau of the stored samples relative to the detected command end
        "sim_tau": sim["t"] - detected_command_end(sim["t"], command),
        "sim_q": np.rad2deg(sim["q"]), "sim_xyz": 1e3 * shifted_marker(sim),
        "enc_tau": enc["t"] - enc["command_duration"], "exp_q": np.column_stack((enc["q1"], enc["q2"])),
        "exp_tau": exp["tau"], "exp_xyz": np.column_stack((exp["x"], exp["y"], exp["z"])),
        # residual record: samples after the exact command end
        "tail": sim["t"] >= t_end,
        "tail_tau": sim["t"] - t_end,
    }
    lo = [rec["sim_tau"].min(), rec["enc_tau"].min(), rec["exp_tau"].min()]
    hi = [rec["sim_tau"].max(), rec["enc_tau"].max(), rec["exp_tau"].max()]
    rec["tau_lo"], rec["tau_hi"] = max([-2.6] + lo), min([2.5] + hi)
    return rec


def motion_metrics(records):
    """RMSE and R^2 of joint angles and absolute tip coordinates over the motion window."""
    rows = []
    for rec in records.values():
        lo, hi = rec["tau_lo"], rec["tau_hi"]
        for j, name in enumerate(("q1", "q2")):
            m = ev.tracking_metric(rec["sim_tau"], rec["sim_q"][:, j], rec["enc_tau"], rec["exp_q"][:, j], lo, hi)
            if np.ptp(rec["exp_q"][:, j]) < 1.0:     # joint held fixed: only absolute errors are meaningful
                m["nrmse_percent"], m["r2"] = np.nan, np.nan
            rows.append([rec["case"], name, "deg", m["rmse"], m["mae"], m["nrmse_percent"], m["r2"]])
        for j, name in enumerate("XYZ"):
            m = ev.tracking_metric(rec["sim_tau"], rec["sim_xyz"][:, j], rec["exp_tau"], rec["exp_xyz"][:, j], lo, hi)
            rows.append([rec["case"], name, "mm", m["rmse"], m["mae"], m["nrmse_percent"], m["r2"]])
    with open(os.path.join(OUT, "motion_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "quantity", "unit", "rmse", "mae", "nrmse_percent", "r2"])
        w.writerows(rows)
    for r in rows:
        print(f"  case {r[0]} {r[1]:2s}: RMSE {r[3]:7.2f} {r[2]}, R2 {r[6]:.4f}, NRMSE {r[5]:.2f} %")


def residuals(rec):
    """Z residual about z_eq = mean over tau in [0, min(20 s, record)] (Section 5.2)."""
    sim_tau, sim_z = rec["tail_tau"][rec["tail"]], rec["sim_xyz"][rec["tail"], 2]
    keep = rec["exp_tau"] >= 0.0
    exp_tau, exp_z = rec["exp_tau"][keep], rec["exp_xyz"][keep, 2]
    common_hi = min(sim_tau.max(), exp_tau.max())
    hi = min(20.0, common_hi)
    sim_r = sim_z - np.mean(sim_z[(sim_tau >= 0) & (sim_tau <= hi)])
    exp_r = exp_z - np.mean(exp_z[(exp_tau >= 0) & (exp_tau <= hi)])
    return sim_tau, sim_r, exp_tau, exp_r, common_hi, hi


def table3(records):
    rows = []
    for rec in records.values():
        sim_tau, sim_r, exp_tau, exp_r, _, hi = residuals(rec)
        row = [rec["case"]]
        for tau, r in ((sim_tau, sim_r), (exp_tau, exp_r)):
            m = ev.window_rms(tau, r, 0.0, hi)
            row += [m["rms"], m["max_abs"]]
        for tau, r in ((sim_tau, sim_r), (exp_tau, exp_r)):
            f, a = ev.residual_spectrum(tau, r, 2.0)
            row.append(ev.top_peaks(f[f <= 10.0], a[f <= 10.0], 1)[0][0])
        for tau, r in ((sim_tau, sim_r), (exp_tau, exp_r)):
            f, a = ev.residual_spectrum(tau, r, hi)        # 0.05 Hz resolution
            row.append(ev.top_peaks(f, a, 1, f_min=1.0)[0][0])
        rows.append(row)
    header = ["case", "rms_sim_mm", "peak_sim_mm", "rms_exp_mm", "peak_exp_mm",
              "f_sim_0to2s_hz", "f_exp_0to2s_hz", "f_sim_0to20s_hz", "f_exp_0to20s_hz"]
    with open(os.path.join(OUT, "table3_residual_z.csv"), "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print("  Table 4 (RMS / peak over tau in [0, 20] s; f over [0, 2] s and [0, 20] s):")
    for r in rows:
        print(f"    case {r[0]}: RMS {r[1]:.2f} / {r[3]:.2f} mm, peak {r[2]:.2f} / {r[4]:.2f} mm, "
              f"f {r[5]:.1f} / {r[6]:.1f} Hz, 20 s: {r[7]:.2f} / {r[8]:.2f} Hz (sim / exp)")


# ----------------------------------------------------------------------------
# Amplitude dependence and command excitation (Section 5.2)
# ----------------------------------------------------------------------------
def fit_decay(tau, r, lo, hi):
    """Decaying sinusoid A exp(-s t) cos(2 pi f t + phi) + c fitted to r over [lo, hi].

    Returns the amplitude at the window start (mm), the frequency (Hz) and the
    damping ratio s / (2 pi f)."""
    k = (tau >= lo) & (tau <= hi)
    t, x = tau[k] - lo, r[k]
    n = 8 * len(x)
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)), n))
    freq = np.fft.rfftfreq(n, float(np.median(np.diff(t))))
    f0 = freq[freq > 1.0][np.argmax(spec[freq > 1.0])]

    def model(t, a, s, f, phi, c):
        return a * np.exp(-s * t) * np.cos(2.0 * np.pi * f * t + phi) + c

    best = None
    for phi in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        try:
            p, _ = curve_fit(model, t, x, p0=[np.sqrt(2.0) * np.std(x), 0.3, f0, phi, 0.0], maxfev=20000)
        except RuntimeError:
            continue
        err = float(np.sum((x - model(t, *p)) ** 2))
        if best is None or err < best[1]:
            best = (p, err)
    a, s, f, _, _ = best[0]
    return abs(a), f, s / (2.0 * np.pi * f)


def unshaped_residual(case, trial):
    """Z residual of one measured unshaped trial about its mean over tau in [0, 20] s."""
    e = load_trial(os.path.join(DATA, f"case_{case}", "none", trial))
    keep = e["tau"] >= 0.0
    tau, z = e["tau"][keep], e["z"][keep]
    hi = min(20.0, tau.max())
    return tau, z - np.mean(z[tau <= hi])


def window_fits(records):
    """Frequency and damping of the Z residual in successive windows: simulation and all unshaped trials."""
    rows = []
    for case, rec in records.items():
        sim_tau, sim_r, _, _, _, _ = residuals(rec)
        sources = [("simulation", sim_tau, sim_r)] + [(f"trial {k}", *unshaped_residual(case, k)) for k in "123"]
        for name, tau, r in sources:
            for lo, hi in WINDOWS:
                if tau.max() >= hi - 0.5:
                    rows.append([case, name, lo, hi, *fit_decay(tau, r, lo, hi)])
    with open(os.path.join(OUT, "residual_window_fits.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "source", "tau_lo_s", "tau_hi_s", "amplitude_mm", "frequency_hz", "damping_ratio"])
        w.writerows(rows)
    print("  Z residual, decaying-sinusoid fits (frequency / damping ratio / amplitude at window start):")
    for case in records:
        for lo, hi in WINDOWS:
            sim = [r for r in rows if r[0] == case and r[1] == "simulation" and r[2] == lo]
            exp = [r for r in rows if r[0] == case and r[1] != "simulation" and r[2] == lo]
            print(f"    case {case} [{lo:.0f}, {hi:.0f}] s: sim {sim[0][5]:.3f} Hz / {sim[0][6]:.4f}; "
                  f"exp {min(r[5] for r in exp):.3f}-{max(r[5] for r in exp):.3f} Hz / "
                  f"{min(r[6] for r in exp):.4f}-{max(r[6] for r in exp):.4f} / "
                  f"{min(r[4] for r in exp):.2f}-{max(r[4] for r in exp):.2f} mm")


def free_amplitude(t, q, f, zeta, t_end):
    """Largest response over one period after t_end of a single mode at f driven by the joint angle q."""
    w = 2.0 * np.pi * f
    _, y, _ = signal.lsim(signal.lti([-1.0, 0.0, 0.0], [1.0, 2.0 * zeta * w, w * w]), q, t - t[0])
    k = (t >= t_end) & (t <= t_end + 1.0 / f)
    return float(np.max(np.abs(y[k])))


def command_excitation():
    """Free vibration left by the commanded joint-2 angle on a single mode, versus the mode frequency."""
    sweep = np.round(np.arange(2.8, 4.0001, 0.01), 2)
    rows, amp = [], {}
    for case in ("A", "B"):
        spec = P.CASES[case]
        command = PointToPoint(spec["q_start_deg"], spec["q_end_deg"], length=1.0)
        t = np.arange(0.0, 10.0, 0.002)
        q2 = np.array([command.joint_pva(tk)[0][1] for tk in t])
        q2 = np.rad2deg(q2 - q2[0])
        t_end = detected_command_end(t, command)
        for f in sweep:
            a = R_TIP_MM * np.deg2rad(free_amplitude(t, q2, f, ZETA_MEASURED, t_end))
            amp[(case, f)] = a
            rows.append([case, f, a])
    with open(os.path.join(OUT, "command_excitation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "mode_frequency_hz", "free_amplitude_mm_at_1.2m"])
        w.writerows(rows)
    print(f"  single mode (zeta = {ZETA_MEASURED}) driven by the commanded joint-2 angle:")
    for case, others in (("A", (3.45, 3.55)), ("B", (3.55, 3.60))):
        band = [f for f in sweep if 3.3 <= f <= 3.9]
        f_min = min(band, key=lambda f: amp[(case, f)])
        print(f"    case {case}: amplitude at 3.10 Hz / at " +
              ", ".join(f"{f:.2f} Hz {amp[(case, 3.10)] / amp[(case, f)]:.1f}" for f in others) +
              f"; minimum in 3.3-3.9 Hz at {f_min:.2f} Hz")


def relative_to_start(tau, values, tau_start):
    order = np.argsort(tau)
    base = np.array([np.interp(tau_start, tau[order], values[order, i]) for i in range(values.shape[1])])
    return values - base


def print_case_b_x_displacement(records):
    """Report the minimum Case B relative X displacement during the comparison."""
    case_b = records["B"]
    lo = min(records["A"]["tau_lo"], case_b["tau_lo"])
    minima = []
    for source in ("sim", "exp"):
        tau = case_b[f"{source}_tau"]
        xyz = relative_to_start(tau, case_b[f"{source}_xyz"], case_b["tau_lo"])
        x = xyz[:, 0]
        keep = np.isfinite(tau) & np.isfinite(x) & (tau >= lo) & (tau <= 0.5)
        minima.append(x[keep].min())
    print(f"  Case B dX minimum: sim {minima[0]:.0f} mm, exp {minima[1]:.0f} mm")


def show_validation(records: dict) -> None:
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    fig.canvas.manager.set_window_title("s06: simulation versus measurement")
    for (case, rec), (motion, tail) in zip(records.items(), axes):
        motion.plot(rec["sim_tau"], rec["sim_xyz"][:, 2], label="Simulation")
        motion.plot(rec["exp_tau"], rec["exp_xyz"][:, 2], label="Measurement")
        motion.set(title=f"Case {case} marker motion", xlim=(rec["tau_lo"], rec["tau_hi"]),
                   ylabel="Marker Z [mm]")
        sim_t, sim_z, exp_t, exp_z, _, hi = residuals(rec)
        tail.plot(sim_t, sim_z, label="Simulation")
        tail.plot(exp_t, exp_z, label="Measurement")
        tail.set(title=f"Case {case} residual", xlim=(0, hi), ylabel="Residual Z [mm]")
        for ax in (motion, tail):
            ax.set_xlabel("Time after command end [s]")
            ax.legend()
    plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_render_argument(p)
    args = p.parse_args()
    records = {case: load_case(case) for case in ("A", "B")}
    motion_metrics(records)
    table3(records)
    window_fits(records)
    command_excitation()
    print_case_b_x_displacement(records)
    if args.render:
        show_validation(records)


if __name__ == "__main__":
    main()
