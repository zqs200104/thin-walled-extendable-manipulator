"""Step 7 - Measured vibration suppression on the prototype (Section 5.3).

Uses only the measurements in data/shaping_tests (2026-11-13): Cases A and B with
the unshaped command and the ZVD shaper designed for 3.1 Hz, the frequency
predicted by the model (damping ratio 0.01), three trials per command. The first
unshaped trial of each case is also the one compared with the model in s06.

Record handling (stemarm.mocap.load_slew): tau = 0 at the end of the command,
when the last joint stops. Each case uses its own baseline, the median over the
last 2 s of the shortest record of that case; J_RMS is evaluated over tau in
[0, 2.5] s and eta_RMS against the mean unshaped J_RMS of the same case.

The same processing is applied to Z and to the horizontal coordinate normal to
the boom at the end pose (Y in Case A, X in Case B). Also reported for the text:
the dominant frequency of every trial over tau in [0, 20] s (0.05 Hz), the
largest residual over tau in [0, 0.5] s, and the RMS during the slew of the
coordinate band-pass filtered to 2.6-4.2 Hz (tau in [-command duration, 0]).

What remains after shaping (Section 5.3): the share of the mean square of the
residual over tau in [0, 2.5] s that lies in the 2.6-4.2 Hz band, and the RMS of
this band over the baseline window (last 2 s). Joint excitation: the commanded and
the encoder angle of each joint (controller log; encoder low-pass filtered at
15 Hz) are applied, up to the end of the command and held afterwards, to a single
mode with the measured frequency (mean over the unshaped trials; Z for joint 2,
horizontal for joint 1) and zeta = 0.013,
    Y/Q = -s^2 / (s^2 + 2 zeta w s + w^2),
and the RMS of the response over tau in [0, 2.5] s is the excitation of the mode
by that joint.

Outputs:
results/s07_ground_shaping/experiment_shaping_trials.csv and
results/s07_ground_shaping/experiment_joint_excitation.csv.
Run time: < 1 min.
"""
import argparse
import csv
import glob
import os

import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt

from stemarm import paths
from stemarm.runner import add_render_argument
from stemarm import evaluation as ev
from stemarm.mocap import load_slew, command_window, _log_path, _read_log

DATA = os.path.join(paths.DATA_DIR, "shaping_tests")
OUT = paths.result_dir("s07_ground_shaping")
SHAPERS = ("none", "zvd_3.1")
HORIZONTAL = {"A": "y", "B": "x"}   # normal to the boom at the end pose: boom along -X (A), -Y (B)
SERIES = (("A", "z"), ("A", "y"), ("B", "z"), ("B", "x"))
T_E = 2.5                 # evaluation window [s]
BASELINE_DURATION = 2.0   # [s]
BAND = (2.6, 4.2)         # Hz, band of the vibration during the slew
ZETA_MEASURED = 0.013     # free-decay tests, Section 4.3
JOINT_LOWPASS_HZ = 15.0


def trial_folders(case, shaper):
    folders = [f for f in glob.glob(os.path.join(DATA, f"case_{case}", shaper, "*")) if os.path.isdir(f)]
    return sorted(folders, key=lambda f: int(os.path.basename(f)))


def band_rms_during_slew(rec, axis):
    tau, v = rec["tau"], rec[axis]
    fs = 1.0 / float(np.median(np.diff(tau)))
    filtered = sosfiltfilt(butter(4, BAND, btype="bandpass", fs=fs, output="sos"), v)
    keep = (tau >= -rec["command_duration"]) & (tau <= 0.0)
    return float(np.sqrt(np.mean(filtered[keep] ** 2)))


def evaluate():
    records = {}
    for case in ("A", "B"):
        for shaper in SHAPERS:
            folders = trial_folders(case, shaper)
            for folder in folders:
                records[(case, shaper, os.path.basename(folder))] = load_slew(
                    folder, siblings=[f for f in folders if f != folder])
    common_hi = {c: min(float(r["tau"].max()) for k, r in records.items() if k[0] == c) for c in ("A", "B")}
    res = {}
    for (case, shaper, trial), rec in records.items():
        hi = common_hi[case]
        for axis in ("z", HORIZONTAL[case]):
            keep = rec["tau"] >= 0.0
            tau, value = rec["tau"][keep], rec[axis][keep]
            residual = value - ev.median_baseline(tau, value, hi - BASELINE_DURATION, hi)
            m20 = tau <= 20.0
            f, a = ev.residual_spectrum(tau, value - np.mean(value[m20]), 20.0)
            # band of the first mode, filtered over the whole record so that tau = 0 is free of edge effects
            full = rec[axis] - (value[0] - residual[0])
            fs = 1.0 / float(np.median(np.diff(rec["tau"])))
            band = sosfiltfilt(butter(4, BAND, btype="bandpass", fs=fs, output="sos"), full)
            w = (rec["tau"] >= 0.0) & (rec["tau"] <= T_E)
            late = (rec["tau"] >= hi - BASELINE_DURATION) & (rec["tau"] <= hi)
            res[(case, axis, shaper, trial)] = {
                "band_share": float(np.mean(band[w] ** 2) / np.mean(full[w] ** 2)),
                "late_band": float(np.sqrt(np.mean(band[late] ** 2))),
                "tau": tau, "residual": residual,
                "rms": ev.window_rms(tau, residual, 0.0, T_E)["rms"],
                "start": float(np.max(np.abs(residual[tau <= 0.5]))),
                "f20": ev.top_peaks(f, a, 1, f_min=1.0)[0][0],
                "during": band_rms_during_slew(rec, axis),
                "duration": rec["command_duration"]}
    return res, common_hi


def statistics(res):
    stats, rows = {}, []
    for case, axis in SERIES:
        trials = {s: sorted({k[3] for k in res if k[:3] == (case, axis, s)}, key=int) for s in SHAPERS}
        J = {s: np.array([res[(case, axis, s, t)]["rms"] for t in trials[s]]) for s in SHAPERS}
        D = {s: np.array([res[(case, axis, s, t)]["during"] for t in trials[s]]) for s in SHAPERS}
        ref = J["none"].mean()
        shaped = J["zvd_3.1"]
        stats[(case, axis)] = {"J": J, "trials": trials, "eta_mean": 100.0 * (1.0 - shaped.mean() / ref),
                               "eta_trials": 100.0 * (1.0 - shaped / ref),
                               "eta_during": 100.0 * (1.0 - D["zvd_3.1"].mean() / D["none"].mean())}
        st = stats[(case, axis)]
        print(f"  case {case} {axis.upper()}: J_RMS unshaped {J['none'].mean():.2f} +- {J['none'].std(ddof=1):.2f}, "
              f"ZVD 3.1 Hz {shaped.mean():.2f} +- {shaped.std(ddof=1):.2f} mm; eta of means {st['eta_mean']:.1f} %, "
              f"single trials {np.round(st['eta_trials'], 1)} %; during slew {st['eta_during']:.1f} %")
        for s in SHAPERS:
            for t in trials[s]:
                r = res[(case, axis, s, t)]
                rows.append([case, axis.upper(), s, t, f"{r['duration']:.3f}", r["rms"],
                             100.0 * (1.0 - r["rms"] / ref), r["f20"], r["start"], r["during"]])
                if s == "none":
                    print(f"    unshaped trial {t}: f over [0, 20] s {r['f20']:.2f} Hz, "
                          f"largest residual over [0, 0.5] s {r['start']:.1f} mm")
    with open(os.path.join(OUT, "experiment_shaping_trials.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "coordinate", "shaper", "trial", "command_duration_s", "J_rms_mm",
                    "eta_rms_vs_mean_unshaped_percent", "f_0to20s_hz", "max_abs_0to0p5s_mm",
                    "band_rms_during_slew_mm"])
        w.writerows(rows)
    return stats


def remaining(res):
    """Share of the shaped residual in the band of the first mode, and the band RMS at the end of the records."""
    shares = [r["band_share"] for k, r in res.items() if k[2] == "zvd_3.1"]
    floors = [r["late_band"] for r in res.values()]
    print(f"  shaped residuals: smallest share of the mean square over [0, {T_E}] s in {BAND[0]}-{BAND[1]} Hz "
          f"{100 * min(shares):.1f} %; band RMS over the last {BASELINE_DURATION:.0f} s of all records "
          f"{min(floors):.3f}-{max(floors):.3f} mm")


def mode_response_rms(t, q, f, t_end):
    """RMS over [t_end, t_end + T_E] of a single mode at f driven by the joint angle q, held after t_end."""
    w = 2.0 * np.pi * f
    mode = signal.lti([-1.0, 0.0, 0.0], [1.0, 2.0 * ZETA_MEASURED * w, w * w])
    tu = np.arange(t[0], t[-1], 0.002)
    qu = np.interp(tu, t, q) - q[0]
    qu[tu > t_end] = qu[np.searchsorted(tu, t_end)]
    _, y, _ = signal.lsim(mode, qu, tu - tu[0])
    keep = (tu >= t_end) & (tu <= t_end + T_E)
    return float(np.sqrt(np.mean(y[keep] ** 2)))


def joint_excitation(res):
    """Excitation of the first mode by the commanded and the measured angle of each joint."""
    cols = ("t_s", "q1_deg", "q2_deg", "q1_ref_deg", "q2_ref_deg")
    freq = {(case, joint): float(np.mean([r["f20"] for k, r in res.items()
                                          if k[0] == case and k[1] == axis and k[2] == "none"]))
            for case in ("A", "B") for joint, axis in (("q2", "z"), ("q1", HORIZONTAL[case]))}
    rows, exc = [], {}
    for case in ("A", "B"):
        for shaper in SHAPERS:
            for folder in trial_folders(case, shaper):
                log = _log_path(folder)
                if not log:
                    continue
                d = _read_log(log, cols)
                t0, duration = command_window(log)
                keep = (d[:, 0] >= t0 - 1.0) & (d[:, 0] <= t0 + duration + 3.0)
                t = d[keep, 0] - t0
                fs = 1.0 / float(np.median(np.diff(t)))
                lowpass = butter(4, JOINT_LOWPASS_HZ, fs=fs, output="sos")
                for joint, im, ir in (("q2", 2, 4), ("q1", 1, 3)):
                    f = freq[(case, joint)]
                    commanded = mode_response_rms(t, d[keep, ir], f, duration)
                    measured = mode_response_rms(t, sosfiltfilt(lowpass, d[keep, im]), f, duration)
                    exc.setdefault((case, joint, shaper), []).append((commanded, measured))
                    rows.append([case, joint, shaper, os.path.basename(folder), f"{f:.2f}", commanded, measured])
    for case in ("A", "B"):
        for joint in (("q2", "q1") if case == "B" else ("q2",)):   # joint 1 does not move in Case A
            u = np.mean(exc[(case, joint, "none")], axis=0)
            z = np.mean(exc[(case, joint, "zvd_3.1")], axis=0)
            print(f"  case {case} joint {joint[1]} at {freq[(case, joint)]:.2f} Hz: unshaped measured/commanded "
                  f"{u[1] / u[0]:.2f}; shaper reduces the commanded excitation by {100 * (1 - z[0] / u[0]):.1f} % "
                  f"and the measured one by {100 * (1 - z[1] / u[1]):.1f} %")
    with open(os.path.join(OUT, "experiment_joint_excitation.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["case", "joint", "shaper", "trial", "mode_frequency_hz",
                     "excitation_commanded_deg", "excitation_measured_deg"])
        wr.writerows(rows)


def show_residuals(res: dict) -> None:
    from matplotlib import pyplot as plt

    fig, panels = plt.subplots(2, 2, figsize=(10, 6), layout="constrained", sharex=True)
    fig.canvas.manager.set_window_title("s07: measured residual vibration")
    for (case, axis), ax in zip(SERIES, panels.flat):
        for shaper, label, color in zip(SHAPERS, ("Unshaped", "ZVD"), ("#D55E00", "#0072B2")):
            trials = [row for key, row in res.items() if key[:3] == (case, axis, shaper)]
            for i, trial in enumerate(trials):
                keep = trial["tau"] <= T_E
                ax.plot(trial["tau"][keep], trial["residual"][keep], color=color, alpha=0.7,
                        label=label if i == 0 else None)
        ax.set(title=f"Case {case}, {axis.upper()}", xlabel="Time after command end [s]",
               ylabel="Measured residual [mm]", xlim=(0, T_E))
        ax.legend()
    plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_render_argument(p)
    args = p.parse_args()
    res, common_hi = evaluate()
    for case in ("A", "B"):
        print(f"  case {case} baseline window: tau in [{common_hi[case] - BASELINE_DURATION:.2f}, "
              f"{common_hi[case]:.2f}] s")
    statistics(res)
    remaining(res)
    joint_excitation(res)
    if args.render:
        show_residuals(res)


if __name__ == "__main__":
    main()
