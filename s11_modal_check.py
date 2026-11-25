"""Step 11 - Modal check of the unshaped ground slews (supports Section 5.2).

Compares three estimates of the first-mode frequency and damping ratio at the
final poses of Cases A and B (L = 1 m):
  * complex modes of the model, without gravity and about the static
    equilibrium under gravity (stemarm.modal.damped_modes);
  * a decaying sinusoid fitted to the simulated residual of s05;
  * the same fit to the first measured trial, in consecutive time windows,
    because the measured frequency and damping change as the amplitude decays.

The results check the spectral values of Section 5.2; no figure or table of the
paper is produced here. Needs the results of s05_ground_slews.py.

Outputs: results/s11_modal_check/complex_modes.csv and residual_fit.csv.
Run time: about 1 min.
"""
import csv
import os

import numpy as np
from scipy.optimize import least_squares

from stemarm import params as P, paths, simulate
from stemarm.mocap import load_trial
from stemarm.modal import damped_modes
from stemarm.runner import parser

OUT = paths.result_dir("s11_modal_check")
DATA = os.path.join(paths.DATA_DIR, "ground_tests")
LENGTH = 1.0                                    # m
WINDOWS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0))   # s, tau
SIMULATION_WINDOW = (0.0, 10.0)                 # s, the simulated residual is tiny later on


def fit_decaying_sine(tau, z, lo, hi, f_start):
    """Fit A exp(-s t) sin(2 pi f t + phi) + c over tau in [lo, hi]; returns f, zeta, A."""
    keep = (tau >= lo) & (tau <= hi) & np.isfinite(z)
    t, y = tau[keep] - lo, z[keep]

    def residual(p):
        with np.errstate(over="ignore", invalid="ignore"):    # trial steps of the optimizer can overflow
            return p[0] * np.exp(-p[1] * t) * np.sin(2.0 * np.pi * p[2] * t + p[3]) + p[4] - y

    best = None
    for phase in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        solution = least_squares(residual, [0.5 * np.ptp(y), 0.3, f_start, phase, np.mean(y)])
        if best is None or solution.cost < best.cost:
            best = solution
    f, decay = best.x[2], best.x[1]
    return float(f), float(decay / (2.0 * np.pi * f)), float(abs(best.x[0]))


def complex_modes():
    rows = []
    for case, spec in P.CASES.items():
        q = np.deg2rad(spec["q_end_deg"])
        for gravity in (False, True):
            modes = damped_modes(q, LENGTH, gravity=gravity, n_modes=2)
            for i, (fn, fd, zeta) in enumerate(modes, start=1):
                rows.append([case, "gravity" if gravity else "no gravity", i, fn, fd, zeta])
                print(f"  Case {case}, {'gravity   ' if gravity else 'no gravity'}, mode {i}: "
                      f"fn {fn:.3f} Hz, fd {fd:.3f} Hz, zeta {zeta:.4f}")
    with open(os.path.join(OUT, "complex_modes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "state", "mode", "fn_hz", "fd_hz", "zeta"])
        w.writerows(rows)


def residual_fit():
    rows = []
    for case in P.CASES:
        r = simulate.load(paths.result_dir("s05_ground_slews", f"case_{case}_none"))
        tau, z = r["t"] - r["meta"]["command_end_s"], 1e3 * r["marker"][:, 2]
        f, zeta, amplitude = fit_decaying_sine(tau, z, *SIMULATION_WINDOW, 3.1)
        rows.append([case, "simulation", *SIMULATION_WINDOW, f, zeta, amplitude])
        print(f"  Case {case}, simulation  tau {SIMULATION_WINDOW}: f {f:.3f} Hz, zeta {zeta:.4f}, A {amplitude:.2f} mm")
        trial = load_trial(os.path.join(DATA, f"case_{case}", "none", "1"))
        tau, z = np.asarray(trial["tau"]), np.asarray(trial["z"], dtype=float)
        for lo, hi in WINDOWS:
            f, zeta, amplitude = fit_decaying_sine(tau, z, lo, hi, 3.45)
            rows.append([case, "experiment", lo, hi, f, zeta, amplitude])
            print(f"  Case {case}, experiment  tau ({lo}, {hi}): f {f:.3f} Hz, zeta {zeta:.4f}, A {amplitude:.2f} mm")
    with open(os.path.join(OUT, "residual_fit.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "source", "tau_start_s", "tau_end_s", "f_hz", "zeta", "amplitude_mm"])
        w.writerows(rows)
    return rows


def show_residual_fits(rows: list) -> None:
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    fig.canvas.manager.set_window_title("s11: residual frequency by fit window")
    for case, ax in zip(P.CASES, axes):
        for source in ("simulation", "experiment"):
            values = np.array([row[2:5] for row in rows if row[:2] == [case, source]])
            lo, hi, frequency = values.T
            ax.errorbar((lo + hi) / 2.0, frequency, xerr=(hi - lo) / 2.0, fmt="o",
                        label=source.capitalize())
        ax.set(title=f"Case {case}", xlabel="Fit window after command end [s]")
        ax.legend()
    axes[0].set_ylabel("Fitted frequency [Hz]")
    plt.show()


def main():
    args = parser(__doc__).parse_args()
    print("  Complex modes at L = 1 m:")
    complex_modes()
    print("  Decaying sinusoid fits of the Z residual:")
    rows = residual_fit()
    if args.render:
        show_residual_fits(rows)


if __name__ == "__main__":
    main()
