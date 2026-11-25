"""Step 3 - Identification of the root damping D (Section 4.3).

Free-decay tests (data/root_identification/free_decay_420g): L = 1 m, a 420 g
tip mass, the boom pulled below its equilibrium and released; marker P was
tracked at 90 Hz. The damping ratio is taken from an exponential fit to the
displacement peaks between 5 and 20 mm (both polarities, averaged).

Because the tests measure tip motion and not the root moment, the release is
simulated with the boom model and K fixed, and D is chosen so that the
simulated damping ratio, evaluated in the same way, equals the measured mean.
By default the script checks the value used in the paper (D = 0.15411);
with --fit it searches D by the secant method.

Run time: about 1 min per simulation.
"""
import argparse
import os

import numpy as np
import exudyn as exu
from exudyn import graphics
from exudyn.itemInterface import (
    NodePoint, ObjectMassPoint, MarkerNodeCoordinate, CoordinateConstraint, VCoordinateConstraint,
    LoadCoordinate, NodePointGround, SensorNode, VObjectMassPoint,
)
from scipy.signal import find_peaks, savgol_filter

from stemarm import params as P, paths, rig
from stemarm.mocap import read_nokov
from stemarm.rendering import live
from stemarm.runner import add_render_argument
from stemarm.section import section_properties

DATA = os.path.join(paths.DATA_DIR, "root_identification", "free_decay_420g")
OUT = paths.result_dir("s03_root_damping")
TIP_MASS = 0.420           # kg
LENGTH = 1.0               # m
PULL = 14.17e-3            # m below the loaded equilibrium in the model
BAND = (5.0, 20.0)         # mm, amplitude band of the peak fit
DURATION, STEP = 10.0, 1e-3


def fit_polarity(t, y, peaks):
    a = np.abs(y[peaks])
    q = peaks[(a >= BAND[0]) & (a <= BAND[1])]
    slope = np.polyfit(t[q] - t[q][0], np.log(np.abs(y[q])), 1)[0]
    fd = 1.0 / np.median(np.diff(t[q]))
    return -slope / np.sqrt((2 * np.pi * fd)**2 + slope**2), fd


def decay(t, y, prominence, start=0):
    """Damping ratio and damped frequency from the peaks of y [mm] in BAND."""
    distance = max(2, int(round(0.12 / np.median(np.diff(t)))))
    pos = find_peaks(y, distance=distance, prominence=prominence)[0]
    neg = find_peaks(-y, distance=distance, prominence=prominence)[0]
    (zp, fp), (zn, fn) = fit_polarity(t, y, pos[pos >= start]), fit_polarity(t, y, neg[neg >= start])
    return 0.5 * (zp + zn), 0.5 * (fp + fn)


def measured():
    """Damping ratio of each measured trial (Z about its final equilibrium)."""
    rows = []
    for trial in (1, 2, 3):
        t, xyz = read_nokov(os.path.join(DATA, f"trial_{trial}", "SinglePoint1.ly"))
        z = savgol_filter(xyz[:, 2], 11, 3)
        y = z - np.median(z[int(0.85 * len(z)):])
        zeta, fd = decay(t, y, prominence=0.5, start=int(np.argmax(y)))   # from the first free positive peak
        rows.append((zeta, fd))
        print(f"  measured trial {trial}: zeta = {zeta:.4f}, f_d = {fd:.2f} Hz")
    zeta = np.array([r[0] for r in rows])
    print(f"  measured zeta = {zeta.mean():.4f} +- {zeta.std(ddof=1):.4f} (mean +- SD)")
    return zeta.mean(), np.mean([r[1] for r in rows])


def simulate_release(d_root, *, render: bool = False):
    """Static equilibrium with the tip mass, pull down by PULL, release; returns (zeta, f_d)."""
    sc = exu.SystemContainer()
    mbs = sc.AddSystem()
    props = section_properties()
    boom = rig.build(mbs, LENGTH, P.N_ELEMENTS, P.K_ROOT, d_root, beam_gravity=True, props=props, with_graphics=render)
    tip_ref = np.array(mbs.GetNode(boom["tip"])["referenceCoordinates"][0:3], dtype=float)
    n_mass = mbs.AddNode(NodePoint(referenceCoordinates=list(tip_ref)))
    mbs.AddObject(ObjectMassPoint(physicsMass=TIP_MASS, nodeNumber=n_mass,
                                  visualization=VObjectMassPoint(graphicsData=[
                                      graphics.Sphere(point=[0, 0, 0], radius=0.018, color=graphics.color.red),
                                  ] if render else [])))
    for c in range(3):              # point mass tied to the tip node
        mbs.AddObject(CoordinateConstraint(markerNumbers=[
            mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=boom["tip"], coordinate=c)),
            mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_mass, coordinate=c))],
            visualization=VCoordinateConstraint(show=False)))
    for c, f in enumerate((0.0, -rig.G * TIP_MASS, 0.0)):
        mbs.AddLoad(LoadCoordinate(markerNumber=mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_mass, coordinate=c)),
                                   load=f))
    m_mass_y = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_mass, coordinate=1))
    m_ground_y = mbs.AddMarker(MarkerNodeCoordinate(
        nodeNumber=mbs.AddNode(NodePointGround(referenceCoordinates=[0, 0, 0])), coordinate=1))
    hold = mbs.AddObject(CoordinateConstraint(markerNumbers=[m_ground_y, m_mass_y], offset=0.0,
                                              activeConnector=False, visualization=VCoordinateConstraint(show=False)))
    sensor = mbs.AddSensor(SensorNode(nodeNumber=n_mass, storeInternal=True,
                                      outputVariableType=exu.OutputVariableType.Position))
    mbs.Assemble()
    static = rig.static_settings(load_steps=100, adaptive=True, rel_tol=1e-6, abs_tol=1e-8, max_iter=200,
                                 stabilizer=1e-8)
    static.staticSolver.minimumStepSize = 1e-8
    static.staticSolver.newton.useModifiedNewton = False
    mbs.SolveStatic(static, updateInitialValues=True)                 # 1: equilibrium with the mass
    y_eq = float(mbs.GetSensorValues(sensor)[1])
    mbs.SetObjectParameter(hold, "offset", y_eq - PULL - float(tip_ref[1]))
    mbs.SetObjectParameter(hold, "activeConnector", True)
    mbs.SolveStatic(static, updateInitialValues=True)                 # 2: pulled down and held
    mbs.SetObjectParameter(hold, "activeConnector", False)            # 3: release
    s = exu.SimulationSettings()
    ti = s.timeIntegration
    ti.numberOfSteps, ti.endTime, ti.verboseMode = int(DURATION / STEP), DURATION, 0
    ti.adaptiveStep, ti.minimumStepSize = True, 1e-7
    ti.generalizedAlpha.computeInitialAccelerations = False
    ti.generalizedAlpha.resetAccelerations = True
    ti.generalizedAlpha.spectralRadius = P.SPECTRAL_RADIUS
    ti.newton.useModifiedNewton = True
    ti.newton.relativeTolerance, ti.newton.absoluteTolerance, ti.newton.maxIterations = 1e-4, 1e-6, 100
    ti.newton.maxModifiedNewtonIterations, ti.newton.maxModifiedNewtonRestartIterations = 8, 7
    ti.newton.modifiedNewtonContractivity = 0.5
    ti.newton.modifiedNewtonJacUpdatePerStep = False
    ti.newton.numericalDifferentiation.minimumCoordinateSize = 1.0
    ti.newton.numericalDifferentiation.relativeEpsilon = 1e-6
    s.linearSolverType = exu.LinearSolverType.EigenSparse
    s.solutionSettings.writeSolutionToFile = False
    s.solutionSettings.sensorsWritePeriod = STEP
    with live(sc, render, title=f"s03: free release, D = {d_root:.5f} N m s/rad"):
        mbs.SolveDynamic(s, solverType=exu.DynamicSolverType.GeneralizedAlpha, updateInitialValues=True)
    data = np.asarray(mbs.GetSensorStoredData(sensor), dtype=float)
    return decay(data[:, 0], 1e3 * (data[:, 2] - y_eq), prominence=0.05)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", action="store_true", help="search D instead of checking the paper value")
    add_render_argument(p)
    args = p.parse_args()
    target, fd_exp = measured()
    d = P.D_ROOT
    zeta, fd = simulate_release(d, render=args.render)
    print(f"  simulation D = {d:.5f}: zeta = {zeta:.6f} (target {target:.6f}), f_d = {fd:.3f} Hz "
          f"(measured {fd_exp:.3f} Hz, {100 * (fd / fd_exp - 1):+.1f} %)")
    if args.fit:
        d_prev, z_prev = d, zeta
        d = d * target / zeta                       # zeta grows almost linearly with D
        for _ in range(6):
            zeta, fd = simulate_release(d, render=args.render)
            print(f"  simulation D = {d:.5f}: zeta = {zeta:.6f}")
            if abs(zeta - target) < 1e-6:
                break
            d, d_prev, z_prev = d + (target - zeta) * (d - d_prev) / (zeta - z_prev), d, zeta
    with open(os.path.join(OUT, "root_damping.txt"), "w", encoding="utf-8") as f:
        f.write(f"measured zeta (mean of 3 trials) = {target:.6f}\nD = {d:.6f} N m s/rad\n"
                f"simulated zeta = {zeta:.6f}\nsimulated f_d = {fd:.4f} Hz, measured f_d = {fd_exp:.4f} Hz\n")


if __name__ == "__main__":
    main()
