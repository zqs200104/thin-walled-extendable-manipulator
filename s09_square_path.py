"""Step 9 - Trajectory motion under microgravity: square path of the boom tip (Section 4.6).

The tip traverses a 1 m square in the plane z = 2 m (centre (-1, 0, 2) m),
counter-clockwise from the corner (-0.5, -0.5, 2) m; edges follow a
trapezoidal speed profile (0.5 m/s, 1 m/s^2) and the tip rests 0.3 s at each
corner (T = 11.2 s). A 2 s approach from q = (10, 0) deg, L = 0.05 m is not
shaped. Gravity is off. Commands (zeta = 0.01):
  unshaped;
  ZVD 1.685 Hz  residual frequency of the unshaped motion (zero crossings over 6 s);
  ZVD 1.492 Hz  close to the median of the frozen-pose modes (text of Section 4.6);
  ZVD 1.250 Hz  peak of the unshaped response during the motion;
  ZVD2 1.35 + 1.60 Hz with the zero-time-delay scheme (t_acc = t_N + 0.02 s);
  ZVD2 1.35 + 1.60 Hz without compression (delayed).
The unshaped, ZVD and compressed ZVD2 commands are repeated with a corner rest
of 1.7 s, long enough for every shaper to stop at the corners (text of Section 4.6).

Outputs: frozen-pose modes (table4_frozen_modes.csv), Table 3
(table5_square_path.csv) and long_rest_square_path.csv in results/s09_square_path/.
Run time: about 5 min per case and 1 min per modal snapshot.
"""
import csv
import json
import os

import numpy as np

from stemarm import paths, shaping, simulate
from stemarm import evaluation as ev
from stemarm.path import SquareCommand, APPROACH_TIME, CORNER_HOLD, DT
from stemarm.runner import parser, run_jobs

ZETA = 0.01
POST_COMMAND_TIME = 10.0        # T_e of Section 4.6
BASELINE_DURATION = 2.0         # W_b = [8, 10] s
COMMANDS = {                    # name: (shaper kind, frequencies, zero-time-delay scheme)
    "unshaped": ("none", (), True),
    "zvd_1p250": ("zvd", (1.25,), True),
    "zvd_1p492": ("zvd", (1.492,), True),
    "zvd_1p685": ("zvd", (1.685,), True),
    "zvd2_compressed": ("zvd2", (1.35, 1.60), True),
    "zvd2_delayed": ("zvd2", (1.35, 1.60), False),
}
LONG_REST = 1.7                 # s, corner rest with k T_h >= t_N for every shaper
LONG_REST_COMMANDS = ("unshaped", "zvd_1p250", "zvd_1p492", "zvd_1p685", "zvd2_compressed")
SNAPSHOTS = ("Corner 1 (start, end)", "Edge I midpoint", "Corner 2", "Edge II midpoint",
             "Corner 3", "Edge III midpoint", "Corner 4", "Edge IV midpoint")
OUT = paths.result_dir("s09_square_path")


def make_command(name, corner_hold=CORNER_HOLD):
    kind, freqs, zero_delay = COMMANDS[name]
    shaper = None if kind == "none" else shaping.make_shaper(kind, freqs, ZETA)
    return SquareCommand(shaper, zero_delay=zero_delay, corner_hold=corner_hold)


def simulate_command(name, force=False, long_rest=False, render: bool = False):
    label = f"long_rest_{name}" if long_rest else name
    folder = paths.result_dir("s09_square_path", label)
    if simulate.is_done(folder) and not (force or render):
        print(f"  {label}: stored result used")
        return
    command = make_command(name, LONG_REST if long_rest else CORNER_HOLD)
    print(f"  {label}: command ends at {command.end_time:.3f} s")
    result = simulate.run(command, command.q_start, command.length_pva(0.0)[0],
                          command.end_time + POST_COMMAND_TIME, gravity=False, joint_drive="constraint",
                          variable_length=True, verbose=0,
                          render=render, render_pause=False, render_tip_trace=True)
    result["desired"] = np.array([command.desired_tip(t) for t in result["t"]])
    check = command.check()
    simulate.save(result, folder, {
        "command": name, "shaper": COMMANDS[name][0], "frequencies_hz": list(COMMANDS[name][1]),
        "zeta": ZETA, "zero_time_delay": COMMANDS[name][2], "command_end_s": command.end_time,
        "approach_time_s": APPROACH_TIME, "shaper_duration_s": command.shaper_duration, "k": command.k,
        "t_acc_s": command.t_acc, "corner_speed_m_s": command.corner_speed(), **check})


# ----------------------------------------------------------------------------
# Frozen-configuration modes along the unshaped path (range and median quoted in Section 4.6)
# ----------------------------------------------------------------------------
def snapshot_poses():
    """Eight unshaped command poses equally spaced in arc length along the rigid path."""
    command = make_command("unshaped")
    times = np.append(np.arange(0.0, command.end_time, DT), command.end_time)
    q = np.array([command._pva(t)[0] for t in times])
    keep = times >= APPROACH_TIME - 1e-12
    xyz = np.array([command.fk(*row) for row in q[keep]])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    poses = []
    for label, target in zip(SNAPSHOTS, np.linspace(0.0, arc[-1], len(SNAPSHOTS), endpoint=False)):
        distance = np.abs(arc - target)
        i = int(np.flatnonzero(distance <= distance.min() + 1e-9)[0])   # ties: the earlier sample
        poses.append({"pose": label, "q1_deg": np.rad2deg(q[keep][i, 0]), "q2_deg": np.rad2deg(q[keep][i, 1]),
                      "L_m": q[keep][i, 2]})
    return poses


def modal_snapshot(index, pose, force=False):
    from stemarm.modal import frozen_modes, dominant_planar_mode
    path = os.path.join(paths.result_dir("s09_square_path", "modes"), f"snapshot_{index}.json")
    if os.path.isfile(path) and not force:
        return
    rows = frozen_modes(np.deg2rad([pose["q1_deg"], pose["q2_deg"]]), pose["L_m"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**pose, "f_hz": dominant_planar_mode(rows), "modes": rows.tolist()}, f, indent=1)


def table4():
    rows = []
    for i in range(len(SNAPSHOTS)):
        with open(os.path.join(OUT, "modes", f"snapshot_{i}.json"), encoding="utf-8") as f:
            rows.append(json.load(f))
    with open(os.path.join(OUT, "table4_frozen_modes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pose", "q1_deg", "q2_deg", "L_m", "f_hz"])
        for r in rows:
            w.writerow([r["pose"], f"{r['q1_deg']:.1f}", f"{r['q2_deg']:.1f}", f"{r['L_m']:.3f}", f"{r['f_hz']:.3f}"])
            print(f"    {r['pose']:22s} q1 {r['q1_deg']:6.1f}  q2 {r['q2_deg']:5.1f}  L {r['L_m']:.3f}  f {r['f_hz']:.3f} Hz")
    f = np.array([r["f_hz"] for r in rows])
    print(f"  Frozen-pose modes: dominant mode {f.min():.2f} to {f.max():.2f} Hz, median {np.median(f):.3f} Hz")


# ----------------------------------------------------------------------------
# Evaluation and Table 3
# ----------------------------------------------------------------------------
def load_case(name):
    r = simulate.load(paths.result_dir("s09_square_path", name))
    meta, t = r["meta"], r["t"]
    start, end = meta["approach_time_s"], meta["command_end_s"]
    motion = (t >= start - 1e-9) & (t <= end + 1e-9)
    error = 1e3 * (r["tip"] - r["desired"])
    tau = t - end
    tail = (tau >= -1e-9) & (tau <= POST_COMMAND_TIME + 1e-9)
    residual = error[tail]
    tau_tail = tau[tail]
    residual = residual - np.median(residual[tau_tail >= tau_tail[-1] - BASELINE_DURATION], axis=0)
    return {"r": r, "meta": meta, "t_motion": t[motion] - t[motion][0], "desired": r["desired"][motion],
            "actual": r["tip"][motion], "error_norm": np.linalg.norm(error[motion], axis=1),
            "tau": tau_tail, "residual": residual,
            "e_rms": ev.window_rms(t[motion], np.linalg.norm(error[motion], axis=1), t[motion][0], t[motion][-1])["rms"],
            "J_rms": ev.window_rms(tau_tail, residual[:, 2], 0.0, POST_COMMAND_TIME)["rms"]}


def motion_spectrum(time, signal):
    """Spectrum of the out-of-plane deviation during the motion (linear trend removed, Hann)."""
    centred = time - np.mean(time)
    values = signal - np.polyval(np.polyfit(centred, signal, 1), centred)
    window = np.hanning(len(values))
    amplitude = 2.0 * np.abs(np.fft.rfft(values * window)) / np.sum(window)
    amplitude[0] *= 0.5
    return np.fft.rfftfreq(len(values), float(np.median(np.diff(time)))), amplitude


def evaluate():
    cases = {name: load_case(name) for name in COMMANDS}
    base = cases["unshaped"]
    r = base["r"]
    keep = r["t"] >= base["meta"]["command_end_s"]
    f_res = ev.zero_crossing_frequency(r["t"][keep], r["tip"][keep, 2], 6.0)
    print(f"  residual frequency of the unshaped motion (zero crossings, 6 s): {f_res:.3f} Hz")
    for name in ("unshaped", "zvd_1p685"):
        c = cases[name]
        dz = 1e3 * (c["actual"][:, 2] - c["desired"][:, 2])
        f, a = motion_spectrum(c["t_motion"], dz)
        k = np.argmin(np.abs(f - 1.25))
        band = (f >= 0.2) & (f <= 5.0)
        print(f"  {name}: max |dz| during the motion {np.max(np.abs(1e3 * c['actual'][:, 2] - 1e3 * np.median(c['desired'][:, 2]))):.1f} mm, "
              f"spectrum at {f[k]:.2f} Hz: {a[k]:.2f} mm, peak at {f[band][np.argmax(a[band])]:.2f} Hz")
    rows = []
    for name, c in cases.items():
        m = c["meta"]
        eta = 100.0 * (1.0 - c["J_rms"] / base["J_rms"])
        rows.append([name, m["shaper_duration_s"], m["k"], m["corner_speed_m_s"], c["e_rms"], c["J_rms"],
                     eta, m["qdd_max"][0], m["contour_max_mm"]])
    with open(os.path.join(OUT, "table5_square_path.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["command", "t_N_s", "k", "v_c_m_s", "e_rms_mm", "J_rms_mm", "eta_rms_percent",
                    "qdd1_max_rad_s2", "rigid_path_max_deviation_mm"])
        w.writerows(rows)
    print("  Table 3:   command           t_N     k     v_c   e_RMS   J_RMS   eta    |qdd1|max  path dev.")
    for r_ in rows:
        print(f"    {r_[0]:16s} {r_[1]:6.3f} {r_[2]:6.3f} {r_[3]:6.3f} {r_[4]:7.2f} {r_[5]:7.3f} {r_[6]:6.1f} "
              f"{r_[7]:8.1f} {r_[8]:8.2f} mm")
    return cases


def long_rest_table():
    """The commands with a corner rest of LONG_REST; eta relative to the unshaped command with the same rest."""
    cases = {name: load_case(f"long_rest_{name}") for name in LONG_REST_COMMANDS}
    base = cases["unshaped"]["J_rms"]
    rows = []
    for name, c in cases.items():
        m = c["meta"]
        rows.append([name, m["shaper_duration_s"], m["k"], m["corner_speed_m_s"], c["e_rms"], c["J_rms"],
                     100.0 * (1.0 - c["J_rms"] / base), m["qdd_max"][0], m["contour_max_mm"]])
    with open(os.path.join(OUT, "long_rest_square_path.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["command", "t_N_s", "k", "v_c_m_s", "e_rms_mm", "J_rms_mm", "eta_rms_percent",
                    "qdd1_max_rad_s2", "rigid_path_max_deviation_mm"])
        w.writerows(rows)
    print(f"  Corner rest {LONG_REST} s: command  t_N     k     v_c   e_RMS   J_RMS   eta    |qdd1|max  path dev.")
    for r_ in rows:
        print(f"    {r_[0]:16s} {r_[1]:6.3f} {r_[2]:6.3f} {r_[3]:6.3f} {r_[4]:7.2f} {r_[5]:7.3f} {r_[6]:6.1f} "
              f"{r_[7]:8.1f} {r_[8]:8.2f} mm")


def main():
    args = parser(__doc__).parse_args()
    jobs = [dict(name=n, force=args.force, render=args.render) for n in COMMANDS]
    jobs += [dict(name=n, force=args.force, long_rest=True, render=args.render) for n in LONG_REST_COMMANDS]
    jobs += [dict(index=i, pose=p, force=args.force) for i, p in enumerate(snapshot_poses())]
    run_jobs(_dispatch, jobs, 1 if args.render else args.workers)
    print("  Frozen-pose modes:")
    table4()
    evaluate()
    long_rest_table()


def _dispatch(**job):
    if "pose" in job:
        modal_snapshot(**job)
    else:
        simulate_command(**job)


if __name__ == "__main__":
    main()
