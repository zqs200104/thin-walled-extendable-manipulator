"""Readers for the prototype measurements and their alignment with the simulation.

* `.ly` files: NOKOV motion-capture export (tab separated, 90 Hz, mm).
* `robot_feedback_*.csv`: controller log with encoder angles, references and torques.

The relative time tau = t - t_end of Eq. (47) is defined by the end of the
commanded motion. For a measurement, t_end = (motion onset seen by the
motion-capture system) + (command duration from the controller log).
"""
import csv
import glob
import os

import numpy as np
from scipy.signal import savgol_filter

from . import params as P


def read_nokov(path):
    """Return time [s] and marker position (N x 3) [mm] of a NOKOV export."""
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                values = [float(parts[i]) for i in (0, 1, 3, 4, 5)]
            except ValueError:
                continue                               # header lines
            if all(np.isfinite(values)):
                rows.append(values)
    data = np.asarray(rows, dtype=float)
    data = data[np.argsort(data[:, 1])]
    return data[:, 1], data[:, 2:5]


def read_encoder(path, q_start_deg=(0.0, 10.0), ref_tol_deg_s=0.05, window=11, polyorder=3):
    """Encoder angles of a slew, smoothed and shifted to the configured start angles.

    The command is taken as active while |dq2_ref| > ref_tol. Some logs stop
    refreshing q2 for a while after the reference stops and then jump to the
    target; that stale interval is bridged with a smooth step.
    Returns t (s, zero at the command start), q1/q2 (deg) and the command
    duration (s).
    """
    cols = ("t_s", "q1_deg", "dq1_deg_s", "q2_deg", "dq2_deg_s", "q2_ref_deg", "dq2_ref_deg_s")
    # a row is used only if all logged joint quantities are numeric
    required = cols + ("e_q2_deg", "tau2_Nm", "q1_ref_deg", "dq1_ref_deg_s", "e_q1_deg", "tau1_Nm")
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                values = {c: float(row[c]) for c in required}
            except (KeyError, TypeError, ValueError):
                continue
            rows.append([values[c] for c in cols])
    data = np.asarray(rows, dtype=float)
    data = data[np.argsort(data[:, 0])]
    t, q1, q2 = data[:, 0], data[:, 1], data[:, 3].copy()
    dq2, q2_ref, dq2_ref = data[:, 4], data[:, 5], data[:, 6]

    active = np.flatnonzero(np.abs(dq2_ref) > ref_tol_deg_s)
    i_start, i_end = int(active[0]), min(int(active[-1]) + 1, len(t) - 1)

    # bridge a stale post-command interval of q2 (see docstring)
    target = float(q2_ref[i_end])
    refreshed = np.flatnonzero(np.abs(q2[i_end:] - target) <= 0.1)
    if len(refreshed):
        i_refresh = i_end + int(refreshed[0])
        if t[i_refresh] - t[i_end] >= 0.5:
            remaining = target - float(q2[i_end])
            speed = max(float(np.median(np.abs(dq2[max(i_start, i_end - 5):i_end + 1]))), 1.0)
            duration = float(np.clip(abs(remaining) / speed, 0.08, 0.40))
            index = np.arange(len(t))
            mask = (index >= i_end) & (index <= i_refresh)
            u = np.clip((t[mask] - t[i_end]) / duration, 0.0, 1.0)
            q2[mask] = q2[i_end] + remaining * u * u * (3.0 - 2.0 * u)
            q2[(t > t[i_end] + duration) & (index < i_refresh)] = target

    window = max(min(window | 1, len(q2) if len(q2) % 2 else len(q2) - 1), (polyorder + 2) | 1)
    q1 = savgol_filter(q1, window, polyorder, mode="interp")
    q2 = savgol_filter(q2, window, polyorder, mode="interp")
    q1 += q_start_deg[0] - q1[i_start]
    q2 += q_start_deg[1] - q2[i_start]
    t_rel = t - t[i_start]
    return {"t": t_rel, "q1": q1, "q2": q2, "command_duration": float(t_rel[i_end])}


def motion_onset(t, xyz, baseline_s=0.5, tol_mm=3.0, hold_s=0.10, fit_mm=(1.0, 30.0), fit_max_s=0.60):
    """Time at which the marker starts to move.

    The marker must leave its initial position by tol_mm and stay out for
    hold_s. Because the displacement grows as (t - t0)^2 during the
    acceleration phase, sqrt(distance) is fitted linearly and extrapolated to
    zero; this removes the delay of the threshold crossing (about 0.06 s).
    """
    dt = float(np.median(np.diff(t)))
    n_base = min(max(3, int(round(baseline_s / dt))), len(t))
    dist = np.linalg.norm(xyz - np.median(xyz[:n_base], axis=0), axis=1)
    n_hold = min(max(1, int(round(hold_s / dt))), len(t))
    moving = dist > tol_mm
    i_cross = next((i for i in range(len(t) - n_hold + 1) if np.all(moving[i:i + n_hold])), None)
    if i_cross is None:
        i_cross = int(np.flatnonzero(moving)[0])
    t_cross = float(t[i_cross])
    n_fit = max(4, int(round(fit_max_s / dt)))
    seg = np.flatnonzero((np.arange(len(dist)) >= i_cross - n_hold)
                         & (dist >= fit_mm[0]) & (dist <= fit_mm[1]))
    if len(seg) < 4:
        return t_cross
    seg = seg[seg <= seg[0] + n_fit]
    if len(seg) < 4:
        return t_cross
    slope, intercept = np.polyfit(t[seg], np.sqrt(dist[seg]), 1)
    if not np.isfinite(slope) or slope <= 0:
        return t_cross
    t_onset = float(-intercept / slope)
    return t_onset if t_cross - 0.5 <= t_onset <= t_cross else t_cross


def load_trial(folder, q_start_deg=(0.0, 10.0), tau_shift=0.0):
    """One measured slew: marker position in the simulation frame and tau.

    Returns tau (s), x/y/z (mm, origin shifted to the simulation frame) and
    the encoder record.
    """
    t, xyz = read_nokov(os.path.join(folder, "SinglePoint0.ly"))
    xyz = xyz - P.MOCAP_ORIGIN_OFFSET_MM
    encoder = read_encoder(sorted(glob.glob(os.path.join(folder, "robot_feedback_*.csv")))[0],
                           q_start_deg=q_start_deg)
    t_end = motion_onset(t, xyz) + encoder["command_duration"]
    return {"tau": t - t_end - tau_shift, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
            "encoder": encoder, "folder": folder}


# ----------------------------------------------------------------------------
# Shaping tests (Section 5.3): command end from both joints, records without a
# resting stretch or without a controller log
MIN_REST_FRAMES = 9    # record frames before the detected onset needed to trust motion_onset (0.1 s)


def _log_path(folder):
    logs = sorted(glob.glob(os.path.join(folder, "robot_feedback_*.csv")))
    return logs[0] if logs else None


def _read_log(path, cols):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append([float(row[c]) for c in cols])
            except (KeyError, TypeError, ValueError):
                continue
    data = np.asarray(rows, dtype=float)
    return data[np.argsort(data[:, 0])]


def command_window(path, ref_tol_deg_s=0.05):
    """Start (controller clock) and duration of the main slew in a controller log.

    The command is active while |dq1_ref| or |dq2_ref| exceeds ref_tol, so the
    command ends when the last joint stops. A log may hold small reference
    corrections before the slew; the main slew is the active segment with the
    largest q2 travel.
    """
    data = _read_log(path, ("t_s", "dq1_ref_deg_s", "dq2_ref_deg_s", "q2_ref_deg"))
    t, d1, d2, q2r = data.T
    active = np.flatnonzero((np.abs(d1) > ref_tol_deg_s) | (np.abs(d2) > ref_tol_deg_s))
    segments = np.split(active, np.flatnonzero(np.diff(active) > 5) + 1)
    seg = max(segments, key=lambda s: abs(q2r[s[-1]] - q2r[s[0]]))
    i0, i1 = int(seg[0]), min(int(seg[-1]) + 1, len(t) - 1)
    return float(t[i0]), float(t[i1] - t[i0])


def _half_travel(folder):
    """Motion-capture time at which the marker has covered half of its start-to-end chord,
    controller time from the slew start to the midpoint of the q2 travel, detected onset and
    number of record frames before the onset."""
    t, xyz = read_nokov(os.path.join(folder, "SinglePoint0.ly"))
    onset = motion_onset(t, xyz)
    p_end = xyz[t > t[-1] - 2.0].mean(axis=0)
    progress = np.linalg.norm(xyz - xyz[0], axis=1) / np.linalg.norm(p_end - xyz[0])
    j = int(np.flatnonzero(progress >= 0.5)[0])
    t_half_mocap = t[j - 1] + (0.5 - progress[j - 1]) / (progress[j] - progress[j - 1]) * (t[j] - t[j - 1])
    t_half_ctrl = None
    log = _log_path(folder)
    if log:
        t0, duration = command_window(log)
        tc, q2 = _read_log(log, ("t_s", "q2_deg")).T
        keep = (tc >= t0 - 0.5) & (tc <= t0 + duration + 0.5)
        tc, q2 = tc[keep], q2[keep]
        t_half_ctrl = float(tc[np.flatnonzero(q2 >= 0.5 * (q2[0] + q2[-1]))[0]] - t0)
    return float(t_half_mocap), t_half_ctrl, onset, int(np.count_nonzero(t < onset))


def load_slew(folder, siblings=()):
    """One measured slew of the shaping tests: marker position in the simulation frame and tau.

    t_end = motion onset + command duration (command_window of the trial's log; for a
    trial without a log, the mean duration of the sibling trials that have one).
    If the motion capture starts less than MIN_REST_FRAMES before the detected onset,
    motion_onset cannot see the start of the motion; the onset is then placed from the
    half-travel time, with the mean onset-to-half-travel offset of the sibling trials
    that have a resting stretch and a log. Siblings are the other trials of the same
    case and command.
    """
    t, xyz = read_nokov(os.path.join(folder, "SinglePoint0.ly"))
    xyz = xyz - P.MOCAP_ORIGIN_OFFSET_MM
    log = _log_path(folder)
    if log:
        duration = command_window(log)[1]
    else:
        durations = [command_window(_log_path(s))[1] for s in siblings if _log_path(s)]
        duration = float(np.mean(durations))
    t_half_m, t_half_c, onset, n_rest = _half_travel(folder)
    from_half_travel = n_rest < MIN_REST_FRAMES
    if from_half_travel:
        offsets = []
        for s in siblings:
            hm, hc, o, n = _half_travel(s)
            if hc is not None and n >= MIN_REST_FRAMES:
                offsets.append((hm - o) - hc)
        if t_half_c is None or not offsets:
            raise RuntimeError(f"cannot place the motion onset of {folder}")
        onset = t_half_m - t_half_c - float(np.mean(offsets))
    t_end = onset + duration
    return {"tau": t - t_end, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "folder": folder,
            "command_duration": duration, "onset_from_half_travel": from_half_travel, "has_log": bool(log)}
