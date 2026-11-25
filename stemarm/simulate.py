"""Time integration and result handling for the manipulator model."""
import json
import os
import time

import numpy as np
import exudyn as exu
from exudyn.rigidBodyUtilities import RotationMatrixX, RotationMatrixZ

from . import params as P
from .model import Manipulator
from .rendering import live


def simulation_settings(end_time, step_size=P.STEP_SIZE, spectral_radius=P.SPECTRAL_RADIUS,
                        modified_newton=True, sensor_period=P.SENSOR_PERIOD, verbose=1):
    """Generalized-alpha settings used for all manipulator simulations of the paper."""
    sim = exu.SimulationSettings()
    ti = sim.timeIntegration
    ti.numberOfSteps = int(end_time / step_size)
    ti.endTime = end_time
    ti.verboseMode = verbose
    ti.adaptiveStep = True           # the step is only reduced if Newton fails
    ti.minimumStepSize = 1.0e-7
    ti.generalizedAlpha.spectralRadius = spectral_radius
    ti.generalizedAlpha.computeInitialAccelerations = False
    ti.generalizedAlpha.resetAccelerations = True
    newton = ti.newton
    newton.useModifiedNewton = bool(modified_newton)
    newton.relativeTolerance = 1.0e-4
    newton.absoluteTolerance = 1.0e-6
    newton.maxIterations = 100
    newton.maxModifiedNewtonIterations = 8
    newton.maxModifiedNewtonRestartIterations = 7
    newton.modifiedNewtonContractivity = 0.5
    newton.modifiedNewtonJacUpdatePerStep = False
    newton.numericalDifferentiation.minimumCoordinateSize = 1.0
    newton.numericalDifferentiation.relativeEpsilon = 1.0e-6
    sim.linearSolverType = exu.LinearSolverType.EigenSparse
    sim.solutionSettings.writeSolutionToFile = False
    sim.solutionSettings.sensorsWritePeriod = sensor_period
    sim.displayComputationTime = verbose > 0
    return sim


def run(command, q0, length0, end_time, gravity=True, joint_drive="constraint",
        variable_length=False, step_size=P.STEP_SIZE, modified_newton=True, render=False,
        verbose=1, spectral_radius=P.SPECTRAL_RADIUS, *, render_pause: bool = True,
        render_tip_trace: bool = False):
    """Build the model, integrate until end_time and return the stored responses.

    Set render_pause=False to start immediately and close after the solve.
    Set render_tip_trace=True to draw the stored tip-node positions when rendering.

    Returns a dict of arrays sampled every SENSOR_PERIOD:
      t, q, qd (joint angles/rates), tip (tip node position), tip_rot (3x3),
      root (root node position), root_rot, theta_root (root rotation),
      marker (marker point P), plus the model object under 'model'.
    """
    sc = exu.SystemContainer()
    mbs = sc.AddSystem()
    model = Manipulator(mbs, command, q0, length0, gravity=gravity, joint_drive=joint_drive,
                        variable_length=variable_length, with_graphics=render)
    mbs.Assemble()
    if render and render_tip_trace:
        traces = sc.visualizationSettings.sensors.traces
        traces.listOfPositionSensors = [int(model.sensors["tip_pos"])]
        traces.showPositionTrace = True
        traces.positionsShowEvery = 10
        traces.lineWidth = 3
        colors = list(traces.traceColors)
        colors[:4] = [0.83, 0.12, 0.15, 1.0]
        traces.traceColors = colors
    sim = simulation_settings(end_time, step_size=step_size, modified_newton=modified_newton,
                              spectral_radius=spectral_radius, verbose=verbose)
    view_rotation = RotationMatrixX(-0.85) @ RotationMatrixZ(0.38) if render else None
    with live(sc, render, title="Stemarm", pause=render_pause, rotation=view_rotation):
        start = time.time()
        mbs.SolveDynamic(sim, solverType=exu.DynamicSolverType.GeneralizedAlpha)
    if verbose:
        print(f"  solved in {time.time() - start:.1f} s")
    return collect(mbs, model)


def collect(mbs, model):
    data = {k: np.asarray(mbs.GetSensorStoredData(s), dtype=float) for k, s in model.sensors.items()}
    n = min(len(v) for v in data.values())
    out = {"t": data["q"][:n, 0], "q": data["q"][:n, 1:3], "qd": data["qd"][:n, 1:3],
           "tip": data["tip_pos"][:n, 1:4], "tip_rot": data["tip_rot"][:n, 1:].reshape(-1, 3, 3),
           "root": data["root_pos"][:n, 1:4], "root_rot": data["root_rot"][:n, 1:].reshape(-1, 3, 3),
           "theta_root": data["theta_root"][:n, 1]}
    out["marker"] = out["tip"] + np.einsum("nij,j->ni", out["tip_rot"], model.marker_offset)
    out["model"] = model
    return out


def save(result, folder, meta):
    """Store a simulation in folder/result.npz (+ meta.json) and a readable CSV."""
    os.makedirs(folder, exist_ok=True)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(os.path.join(folder, "result.npz"), **arrays)
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    table = np.column_stack((result["t"], np.rad2deg(result["q"]), 1e3 * result["tip"],
                             1e3 * result["marker"], np.rad2deg(result["theta_root"])))
    np.savetxt(os.path.join(folder, "response.csv"), table, delimiter=",", fmt="%.9g", comments="",
               header="t_s,q1_deg,q2_deg,tip_x_mm,tip_y_mm,tip_z_mm,"
                      "marker_x_mm,marker_y_mm,marker_z_mm,root_rotation_deg")


def load(folder):
    """Inverse of save(): arrays as a dict plus the metadata under 'meta'."""
    with np.load(os.path.join(folder, "result.npz")) as f:
        out = {k: f[k] for k in f.files}
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        out["meta"] = json.load(f)
    return out


def is_done(folder):
    return os.path.isfile(os.path.join(folder, "result.npz"))
