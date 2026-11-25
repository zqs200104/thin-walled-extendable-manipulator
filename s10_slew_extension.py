"""Step 10 - Slew combined with boom extension (Section 4.5).

The Case B slew (q1 0 -> 90 deg, q2 10 -> 90 deg) is combined with an
extension from 1.0 m to 1.2 m at 50 mm/s and 50 mm/s^2, gravity on. All three
axes are synchronised to the extension time (5.0 s). Commands:
  unshaped; ZVD at 3.10 Hz (residual frequency of the fixed-length Case B slew,
  L = 1.0 m, from s05); ZVD at 2.55 Hz (residual frequency after the unshaped
  combined motion, L = 1.2 m); ZVD2 at both. zeta = 0.01; the shaper acts on
  q1, q2 and L, so all axes still arrive together.

J_RMS uses the Z residual of the tip node about the median over tau in
[15, 20] s, evaluated over tau in [0, 2.5] s. The same commands are repeated
without gravity for the comparison in Section 4.5.

Needs: s05_ground_slews.py (Case B unshaped, for the 3.10 Hz reference).
Results: results/s10_slew_extension/<command>/ and no_gravity_<command>/ ;
run time about 1 min per case.
"""
import json
import os

import numpy as np

from stemarm import params as P, paths, shaping, simulate
from stemarm.commands import PointToPoint
from stemarm.runner import parser, run_jobs

L_START, L_END = 1.0, 1.2          # m
L_VMAX, L_AMAX = 0.05, 0.05        # m/s, m/s^2 (deployment mechanism of the prototype)
F_INITIAL, F_FINAL = 3.10, 2.55    # Hz, see docstring (both on the 0.05 Hz grid)
ZETA = 0.01
POST_COMMAND_TIME = 20.0
T_E, W_B = 2.5, (15.0, 20.0)
COMMANDS = {                        # name: (shaper, frequencies)
    "unshaped": ("none", ()),
    "zvd_initial": ("zvd", (F_INITIAL,)),
    "zvd_final": ("zvd", (F_FINAL,)),
    "zvd2": ("zvd2", (F_FINAL, F_INITIAL)),
}
OUT = paths.result_dir("s10_slew_extension")


def make_command(name):
    kind, freqs = COMMANDS[name]
    spec = P.CASES["B"]
    return PointToPoint(spec["q_start_deg"], spec["q_end_deg"], shaper=shaping.make_shaper(kind, freqs, ZETA),
                        length=L_START, length_end=L_END, length_vmax=L_VMAX, length_amax=L_AMAX)


def run_folder(name, gravity=True):
    return paths.result_dir("s10_slew_extension", name if gravity else f"no_gravity_{name}")


def simulate_command(name, force=False, gravity=True, render: bool = False):
    folder = run_folder(name, gravity)
    label = name if gravity else f"{name}, no gravity"
    if simulate.is_done(folder) and not (force or render):
        print(f"  {label}: stored result used")
        return
    command = make_command(name)
    print(f"  {label}: command ends at {command.end_time:.3f} s")
    result = simulate.run(command, command.q_start, command.length_pva(0.0)[0],
                          command.end_time + POST_COMMAND_TIME, gravity=gravity, joint_drive="constraint",
                          variable_length=True, verbose=0,
                          render=render, render_pause=False)
    simulate.save(result, folder, {"command": name, "shaper": COMMANDS[name][0],
                                   "frequencies_hz": list(COMMANDS[name][1]), "zeta": ZETA, "gravity": gravity,
                                   "command_end_s": command.end_time, "length_m": [L_START, L_END]})


def residual(result, t_end):
    """tau and Z residual of the tip node [mm] after the command end."""
    keep = result["t"] >= t_end
    tau, z = result["t"][keep] - t_end, 1e3 * result["tip"][keep, 2]
    return tau, z - np.median(z[(tau >= W_B[0] - 1e-9) & (tau <= W_B[1] + 1e-9)])


def spectrum(tau, z, window=POST_COMMAND_TIME):
    keep = tau <= window + 1e-9
    values = z[keep] - z[keep].mean()
    hann = np.hanning(values.size)
    return np.fft.rfftfreq(values.size, tau[1] - tau[0]), 2.0 * np.abs(np.fft.rfft(values * hann)) / hann.sum()


def j_rms(tau, z):
    keep = tau <= T_E + 1e-9
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(np.sqrt(trapezoid(z[keep]**2, tau[keep]) / (tau[keep][-1] - tau[keep][0])))


def evaluate():
    runs = {}
    for name in COMMANDS:
        result = simulate.load(paths.result_dir("s10_slew_extension", name))
        tau, z = residual(result, result["meta"]["command_end_s"])
        f, a = spectrum(tau, z)
        band = f > 0.5
        runs[name] = dict(result=result, tau=tau, z=z, f=f, a=a, J=j_rms(tau, z),
                          f_peak=float(f[band][np.argmax(a[band])]))
    fixed = simulate.load(paths.result_dir("s05_ground_slews", "case_B_none"))
    tau, z = residual(fixed, fixed["meta"]["command_end_s"])
    f, a = spectrum(tau, z)
    runs["fixed"] = dict(tau=tau, z=z, f=f, a=a, f_peak=float(f[f > 0.5][np.argmax(a[f > 0.5])]))
    base = runs["unshaped"]["J"]
    metrics = {}
    for name in COMMANDS:
        run = runs[name]
        run["eta"] = 100.0 * (1.0 - run["J"] / base)
        metrics[name] = {"J_rms_mm": run["J"], "eta_percent": run["eta"], "residual_peak_hz": run["f_peak"],
                         "shaper_duration_s": shaping.duration(make_command(name).shaper)}
        print(f"  {name:12s}: J_RMS = {run['J']:.3f} mm, eta = {run['eta']:5.1f} %, "
              f"t_N = {metrics[name]['shaper_duration_s']:.3f} s")
    print(f"  design frequencies: unshaped combined motion peaks at {runs['unshaped']['f_peak']:.3f} Hz "
          f"(F_FINAL = {F_FINAL}), fixed-length slew at {runs['fixed']['f_peak']:.3f} Hz (F_INITIAL = {F_INITIAL})")
    metrics["no_gravity"] = {}
    reference = None
    for name in COMMANDS:
        result = simulate.load(run_folder(name, gravity=False))
        tau, z = residual(result, result["meta"]["command_end_s"])
        j = j_rms(tau, z)
        reference = reference or j
        eta = 100.0 * (1.0 - j / reference)
        metrics["no_gravity"][name] = {"J_rms_mm": j, "eta_percent": eta}
        print(f"  no gravity, {name:12s}: J_RMS = {j:.4f} mm, eta = {eta:6.2f} %")
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return runs


def main():
    args = parser(__doc__).parse_args()
    run_jobs(simulate_command, [dict(name=n, force=args.force, render=args.render, gravity=g) for g in (True, False) for n in COMMANDS],
             1 if args.render else args.workers)
    evaluate()


if __name__ == "__main__":
    main()
