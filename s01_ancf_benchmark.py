"""Step 1 - Verification of the variable-length boom model (Section 4.2, Fig. 7).

Deployment benchmark of Park et al. (2013), Fig. 3: a beam with initial length
1.8 m and an initial tip deflection of 5 mm is extended at 0.1 m/s without
gravity. Two Exudyn models are compared with the digitized published response:
  * the present model: 3D ANCF beam elements whose reference length follows
    the deployed length (ObjectANCFBeam, transport terms omitted);
  * a planar ALE-ANCF model (ObjectALEANCFCable2D) that keeps the transport
    terms: material enters through the root element, which is split whenever
    it has doubled its length.
Both use 48 elements, h = 0.1 ms and a spectral radius of 0.95. The script
also repeats the benchmark at 0.3 m/s (1.8 m to 2.8 m), the highest extension
rate of the square path of Section 4.6.

Reference: S. Park, H.H. Yoo, J. Chung, Vibrations of an axially moving beam
with deployment or retraction, AIAA J. 51 (2013) 686-696.
Run time: about 2 min (the four simulations in parallel).
"""
import csv
import os

import numpy as np
import exudyn as exu
from exudyn.itemInterface import (
    NodePointGround, MarkerNodeCoordinate, CoordinateConstraint, VCoordinateConstraint, NodePoint2DSlope1,
    NodeGenericODE2, ObjectALEANCFCable2D, VObjectALEANCFCable2D, NodePointSlope23, ObjectANCFBeam,
    VANCFBeam, SensorNode,
)

from stemarm import paths, plotstyle
from stemarm.rendering import live
from stemarm.runner import parser, run_jobs

# Beam of Park et al. (2013). Only A and the in-plane I are given; the other 3D
# section properties follow from an equivalent rectangle with the same A and I.
AREA, I_IN_PLANE = 1.4661e-3, 1.1073e-8
E_MOD, NU, RHO, K_SHEAR = 6.8335e10, 0.3, 2738.6, 0.9
TIP_DEFLECTION = 5e-3
N_ELEMENTS = 48
TIME_STEP, SENSOR_PERIOD, SPECTRAL_RADIUS = 1e-4, 1e-3, 0.95
CASES = {"0.1": {"speed": 0.1, "L_initial": 1.8, "L_total": 3.657, "t_end": 10.0},   # Fig. 3 of Park et al.
         "0.3": {"speed": 0.3, "L_initial": 1.8, "L_total": 2.8, "t_end": 10.0}}     # speed check
OUT = paths.result_dir("s01_ancf_benchmark")
PARK_CURVE = os.path.join(paths.DATA_DIR, "park2013", "fig3_tip_displacement_digitized.csv")


def section():
    height = np.sqrt(12.0 * I_IN_PLANE / AREA)
    width = AREA / height
    i_yy = height * width**3 / 12.0
    a, b = max(width, height), min(width, height)
    torsion = a * b**3 * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - b**4 / (12.0 * a**4)))
    g_mod = E_MOD / (2.0 * (1.0 + NU))
    return {"rhoA": RHO * AREA, "EA": E_MOD * AREA, "EI": E_MOD * I_IN_PLANE, "GA": g_mod * AREA * K_SHEAR,
            "GJ": g_mod * torsion, "EIy": E_MOD * i_yy, "EIz": E_MOD * I_IN_PLANE,
            "rhoIyy": RHO * i_yy, "rhoIzz": RHO * I_IN_PLANE, "rhoJ": RHO * (i_yy + I_IN_PLANE)}


SECTION = section()


def length(t, case):
    return min(case["L_initial"] + case["speed"] * max(0.0, float(t)), case["L_total"])


def speed(t, case):
    return case["speed"] if length(t, case) < case["L_total"] else 0.0


def initial_y(x, case):
    xi = x / case["L_initial"]
    return 0.5 * TIP_DEFLECTION * (3.0 * xi**2 - xi**3)


def initial_slope(x, case):
    xi = x / case["L_initial"]
    return TIP_DEFLECTION / case["L_initial"] * (3.0 * xi - 1.5 * xi**2)


def settings(duration):
    s = exu.SimulationSettings()
    ti = s.timeIntegration
    ti.numberOfSteps = max(1, int(np.ceil(duration / TIME_STEP)))
    ti.endTime = duration
    ti.verboseMode = 0
    ti.generalizedAlpha.spectralRadius = SPECTRAL_RADIUS
    ti.generalizedAlpha.computeInitialAccelerations = False
    ti.newton.useModifiedNewton = True
    ti.newton.relativeTolerance = 1e-6
    ti.newton.absoluteTolerance = 1e-8
    ti.newton.numericalDifferentiation.relativeEpsilon = 1e-6
    ti.newton.maxIterations = 100
    ti.reuseConstantMassMatrix = False
    s.solutionSettings.writeSolutionToFile = False
    s.solutionSettings.sensorsWritePeriod = SENSOR_PERIOD
    s.linearSolverType = exu.LinearSolverType.EigenSparse
    return s


def clamp(mbs, ground_marker, node, n):
    for c in range(n):
        m = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=node, coordinate=c))
        mbs.AddObject(CoordinateConstraint(markerNumbers=[ground_marker, m], visualization=VCoordinateConstraint(show=False)))


def tip_data(mbs, sensor):
    data = np.asarray(mbs.GetSensorStoredData(sensor), dtype=float)
    if data.shape[1] == 3:                                       # 2D node: add z = 0
        data = np.column_stack((data, np.zeros(len(data))))
    return data[:, :4]


# ----------------------------------------------------------------------------
# Present model: 3D ANCF beam with variable element length
# ----------------------------------------------------------------------------
def directors(slope):
    tangent = np.array([1.0, slope, 0.0])
    tangent /= np.linalg.norm(tangent)
    ey = np.cross([0.0, 0.0, 1.0], tangent)
    ey /= np.linalg.norm(ey)
    ez = np.cross(tangent, ey)
    ez /= np.linalg.norm(ez)
    return ey.tolist() + ez.tolist()


def run_beam(case, *, render: bool = False):
    sc = exu.SystemContainer()
    mbs = sc.AddSystem()
    ground_marker = mbs.AddMarker(MarkerNodeCoordinate(
        nodeNumber=mbs.AddNode(NodePointGround(referenceCoordinates=[0, 0, 0])), coordinate=0))
    sec = exu.BeamSection()
    sec.stiffnessMatrix = np.diag([SECTION["EA"], SECTION["GA"], SECTION["GA"], SECTION["GJ"], SECTION["EIy"],
                                   SECTION["EIz"]])
    sec.inertia = np.diag([SECTION["rhoJ"], SECTION["rhoIyy"], SECTION["rhoIzz"]])
    sec.massPerLength = SECTION["rhoA"]
    beam_view = VANCFBeam()
    if render:
        geometry = exu.BeamSectionGeometry()
        height = np.sqrt(12.0 * I_IN_PLANE / AREA)
        width = AREA / height
        corners = exu.Vector2DList()
        for y, z in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)):
            corners.Append([y * height / 2.0, z * width / 2.0])
        geometry.polygonalPoints = corners
        beam_view = VANCFBeam(sectionGeometry=geometry, color=[0.10, 0.25, 0.65, 1.0])
    l0 = case["L_initial"] / N_ELEMENTS
    root = mbs.AddNode(NodePointSlope23(referenceCoordinates=[0.0, initial_y(0.0, case), 0.0]
                                        + directors(initial_slope(0.0, case))))
    previous, elements = root, []
    for k in range(N_ELEMENTS):
        x = (k + 1) * l0
        node = mbs.AddNode(NodePointSlope23(referenceCoordinates=[x, initial_y(x, case), 0.0]
                                            + directors(initial_slope(x, case))))
        elements.append(mbs.AddObject(ObjectANCFBeam(nodeNumbers=[previous, node], physicsLength=l0, sectionData=sec,
                                                     crossSectionPenaltyFactor=[0.05, 0.05, 0.05],
                                                     visualization=beam_view)))
        previous = node
    clamp(mbs, ground_marker, root, 9)
    sensor = mbs.AddSensor(SensorNode(nodeNumber=previous, outputVariableType=exu.OutputVariableType.Position,
                                      storeInternal=True))

    def pre_step(system, t):
        for element in elements:
            system.SetObjectParameter(element, "physicsLength", float(length(t, case) / N_ELEMENTS))
        return True

    mbs.SetPreStepUserFunction(pre_step)
    mbs.Assemble()
    with live(sc, render, title=f"s01: 3D ANCF benchmark at {case['speed']:g} m/s"):
        exu.SolveDynamic(mbs, settings(case["t_end"]))
    return tip_data(mbs, sensor)


# ----------------------------------------------------------------------------
# Planar ALE-ANCF model with material inflow through the root element
# ----------------------------------------------------------------------------
def hermite_state(q0, q1, element_length, s):
    xi = s / element_length
    n = (1 - 3 * xi**2 + 2 * xi**3, element_length * (xi - 2 * xi**2 + xi**3), 3 * xi**2 - 2 * xi**3,
         element_length * (-xi**2 + xi**3))
    d = ((-6 * xi + 6 * xi**2) / element_length, 1 - 4 * xi + 3 * xi**2, (6 * xi - 6 * xi**2) / element_length,
         -2 * xi + 3 * xi**2)
    return np.array([n[0] * q0[0] + n[1] * q0[2] + n[2] * q1[0] + n[3] * q1[2],
                     n[0] * q0[1] + n[1] * q0[3] + n[2] * q1[1] + n[3] * q1[3],
                     d[0] * q0[0] + d[1] * q0[2] + d[2] * q1[0] + d[3] * q1[2],
                     d[0] * q0[1] + d[1] * q0[3] + d[2] * q1[1] + d[3] * q1[3]])


def ale_segment(states, velocities, lengths, t_start, duration, case):
    """Integrate one segment between two splits of the root element."""
    sc = exu.SystemContainer()
    mbs = sc.AddSystem()
    ground_marker = mbs.AddMarker(MarkerNodeCoordinate(
        nodeNumber=mbs.AddNode(NodePointGround(referenceCoordinates=[0, 0, 0])), coordinate=0))
    nodes = [mbs.AddNode(NodePoint2DSlope1(referenceCoordinates=q.tolist(), initialCoordinates=[0.0] * 4,
                                           initialVelocities=qd.tolist())) for q, qd in zip(states, velocities)]
    elements, ale_constraints = [], []
    for k, element_length in enumerate(lengths):
        ale_speed = speed(t_start, case) if k == 0 else 0.0     # only the root element is fed
        ale_node = mbs.AddNode(NodeGenericODE2(numberOfODE2Coordinates=1, referenceCoordinates=[0.0],
                                               initialCoordinates=[0.0], initialCoordinates_t=[ale_speed]))
        ale_marker = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=ale_node, coordinate=0))
        ale_constraints.append(mbs.AddObject(CoordinateConstraint(
            markerNumbers=[ground_marker, ale_marker], offset=ale_speed, velocityLevel=True, activeConnector=True,
            visualization=VCoordinateConstraint(show=False))))
        elements.append(mbs.AddObject(ObjectALEANCFCable2D(
            physicsLength=float(element_length), physicsMassPerLength=SECTION["rhoA"], physicsMovingMassFactor=1.0,
            physicsBendingStiffness=SECTION["EI"], physicsAxialStiffness=SECTION["EA"], physicsBendingDamping=0.0,
            physicsAxialDamping=0.0, physicsReferenceAxialStrain=0.0, physicsReferenceCurvature=0.0,
            physicsUseCouplingTerms=True, physicsAddALEvariation=True, useReducedOrderIntegration=0,
            strainIsRelativeToReference=0.0, nodeNumbers=[nodes[k], nodes[k + 1], ale_node],
            visualization=VObjectALEANCFCable2D(drawHeight=0.02))))
    clamp(mbs, ground_marker, nodes[0], 4)
    sensor = mbs.AddSensor(SensorNode(nodeNumber=nodes[-1], outputVariableType=exu.OutputVariableType.Position,
                                      storeInternal=True))
    root_length = lengths[0]

    def pre_step(system, t):
        system.SetObjectParameter(elements[0], "physicsLength",
                                  float(root_length + length(t_start + t, case) - length(t_start, case)))
        system.SetObjectParameter(ale_constraints[0], "offset", speed(t_start + t, case))
        return True

    mbs.SetPreStepUserFunction(pre_step)
    mbs.Assemble()
    exu.SolveDynamic(mbs, settings(duration))
    states = [np.asarray(mbs.GetNodeOutput(n, exu.OutputVariableType.CoordinatesTotal), dtype=float) for n in nodes]
    velocities = [np.asarray(mbs.GetNodeOutput(n, exu.OutputVariableType.Coordinates_t), dtype=float) for n in nodes]
    return tip_data(mbs, sensor), states, velocities


def run_ale(case):
    l0 = case["L_initial"] / N_ELEMENTS
    states = [np.array([k * l0, initial_y(k * l0, case), 1.0, initial_slope(k * l0, case)])
              for k in range(N_ELEMENTS + 1)]
    velocities = [np.zeros(4) for _ in states]
    lengths = [l0] * N_ELEMENTS
    t_now, t_end, history = 0.0, case["t_end"], []
    while t_now < t_end - 1e-14:
        needed = 2.0 * l0 - lengths[0]                      # split when the root element has doubled
        will_split = needed <= length(t_end, case) - length(t_now, case) + 1e-12
        segment_end = t_end
        if will_split:                                      # bisection for the split time
            lo, hi = t_now, t_end
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if length(mid, case) - length(t_now, case) < needed else (lo, mid)
            segment_end = hi
        duration = segment_end - t_now
        data, states, velocities = ale_segment(states, velocities, lengths, t_now, duration, case)
        data[:, 0] += t_now
        history.append(data[1:] if history else data)
        lengths[0] += length(segment_end, case) - length(t_now, case)
        t_now += duration
        if will_split and t_now < t_end - 1e-14:
            old = lengths[0]
            states = [states[0].copy(), hermite_state(states[0], states[1], old, l0), states[1].copy()] + \
                     [q.copy() for q in states[2:]]
            velocities = [velocities[0].copy(), hermite_state(velocities[0], velocities[1], old, l0),
                          velocities[1].copy()] + [v.copy() for v in velocities[2:]]
            lengths = [l0, old - l0] + list(lengths[1:])
    return np.vstack(history)


def simulate_case(speed_name, model, force=False, render: bool = False):
    path = os.path.join(OUT, f"tip_{model}_{speed_name}_m_s.csv")
    live_beam = render and model == "beam"
    if os.path.isfile(path) and not (force or live_beam):
        return
    data = run_beam(CASES[speed_name], render=live_beam) if model == "beam" else run_ale(CASES[speed_name])
    np.savetxt(path, data, delimiter=",", header="time_s,tip_x_m,tip_y_m,tip_z_m", comments="", fmt="%.12g")


def local_frequency(time_s, displacement, case):
    """Local frequency over the full cycle around each internal zero crossing."""
    time_s = np.asarray(time_s, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    indices = np.flatnonzero(displacement[:-1] * displacement[1:] < 0.0)
    x0, x1 = displacement[indices], displacement[indices + 1]
    t0, t1 = time_s[indices], time_s[indices + 1]
    crossings = t0 - x0 * (t1 - t0) / (x1 - x0)
    if len(crossings) < 3:
        raise RuntimeError("At least three zero crossings are required for local frequency")
    center = crossings[1:-1]
    frequency = 1.0 / (crossings[2:] - crossings[:-2])
    deployed_length = np.array([length(t, case) for t in center])
    return deployed_length, frequency, center


def evaluate_and_plot(*, render: bool = False):
    load = lambda m, s: np.loadtxt(os.path.join(OUT, f"tip_{m}_{s}_m_s.csv"), delimiter=",", skiprows=1)
    rms = lambda v: float(np.sqrt(np.mean(v**2)))
    lines = []
    for s in CASES:
        beam, ale = load("beam", s), load("ale", s)
        diff = 1e3 * (np.interp(beam[:, 0], ale[:, 0], ale[:, 2]) - beam[:, 2])
        lines.append(f"{s} m/s: RMS difference ALE-ANCF minus present model {rms(diff):.2f} mm")
    beam, ale = load("beam", "0.1"), load("ale", "0.1")
    park = np.loadtxt(PARK_CURVE, delimiter=",", skiprows=1)
    dev_beam = np.interp(park[:, 0], beam[:, 0], 1e3 * beam[:, 2]) - park[:, 1]
    dev_ale = np.interp(park[:, 0], ale[:, 0], 1e3 * ale[:, 2]) - park[:, 1]
    park_length, park_frequency, park_crossing = local_frequency(park[:, 0], park[:, 1], CASES["0.1"])
    beam_length, beam_frequency, beam_crossing = local_frequency(beam[:, 0], beam[:, 2], CASES["0.1"])
    ale_length, ale_frequency, ale_crossing = local_frequency(ale[:, 0], ale[:, 2], CASES["0.1"])
    beam_frequency_error = (np.abs(np.interp(park_length, beam_length, beam_frequency) - park_frequency)
                            / park_frequency)
    ale_frequency_error = (np.abs(np.interp(park_length, ale_length, ale_frequency) - park_frequency)
                           / park_frequency)
    lines.append(f"0.1 m/s: RMS deviation from Park et al.: present model {rms(dev_beam):.2f} mm, "
                 f"ALE-ANCF {rms(dev_ale):.2f} mm")
    lines.append(f"0.1 m/s: local frequency Park et al. {park_frequency[0]:.3f}-{park_frequency[-1]:.3f} Hz, "
                 f"present model {beam_frequency[0]:.3f}-{beam_frequency[-1]:.3f} Hz, "
                 f"ALE-ANCF {ale_frequency[0]:.3f}-{ale_frequency[-1]:.3f} Hz")
    lines.append(f"0.1 m/s: maximum local-frequency difference from Park et al.: "
                 f"present model {100.0 * beam_frequency_error.max():.2f}%, "
                 f"ALE-ANCF {100.0 * ale_frequency_error.max():.2f}%")
    with open(os.path.join(OUT, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("  " + "\n  ".join(lines))
    np.savetxt(os.path.join(OUT, "local_frequency_vs_length.csv"),
               np.column_stack((park_length, park_frequency, park_crossing,
                                beam_length, beam_frequency, beam_crossing,
                                ale_length, ale_frequency, ale_crossing)),
               delimiter=",", comments="", fmt="%.12g",
               header="reference_length_m,reference_frequency_hz,reference_crossing_time_s,"
                      "present_length_m,present_frequency_hz,present_crossing_time_s,"
                      "ale_length_m,ale_frequency_hz,ale_crossing_time_s")

    W, H = 83.7, 106.5                                   # single-column width; original panel sizes retained
    fig = plotstyle.figure(W, H)
    left, right = 15.66 / W, 1 - 2.0 / W
    ax_a = fig.add_axes([left, 69.5 / H, right - left, 32.0 / H])
    ax_b = fig.add_axes([left, 39.0 / H, right - left, 20.0 / H])
    ax_c = fig.add_axes([left, 8.5 / H, right - left, 20.0 / H])
    ax_a.plot(beam[:, 0], 1e3 * beam[:, 2], color=plotstyle.SIM, lw=0.9, label="Present 3D ANCF", zorder=3)
    ax_a.plot(ale[:, 0], 1e3 * ale[:, 2], color=plotstyle.EXP, lw=0.8, label="Planar ALE-ANCF", zorder=2)
    ax_a.plot(park[:, 0], park[:, 1], color=plotstyle.BLACK, lw=0.9, ls=(0, (3.0, 1.6)), label="Park et al. (ref.)",
              zorder=4)
    ax_a.set(xlim=(0, 10), ylim=(-9.5, 12.5), yticks=[-8, -4, 0, 4, 8], ylabel="Tip displacement [mm]")
    ax_a.tick_params(labelbottom=False)
    ax_a.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
                handlelength=2.2, columnspacing=1.0, borderaxespad=0.2, labelspacing=0.25)
    ax_b.axhline(0.0, color="0.6", lw=0.5, zorder=0)
    ax_b.plot(park[:, 0], dev_beam, color=plotstyle.SIM, lw=0.8, label=f"RMS {rms(dev_beam):.2f} mm")
    ax_b.plot(park[:, 0], dev_ale, color=plotstyle.EXP, lw=0.8, label=f"RMS {rms(dev_ale):.2f} mm")
    lim = 1.15 * max(np.abs(dev_beam).max(), np.abs(dev_ale).max())
    ax_b.set(xlim=(0, 10), ylim=(-lim, 2.25 * lim), xlabel="Time [s]", ylabel="Deviation [mm]")
    ax_b.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
                handlelength=2.2, borderaxespad=0.15, labelspacing=0.25)
    ax_c.plot(beam_length, beam_frequency, color=plotstyle.SIM, lw=0.9, zorder=3)
    ax_c.plot(park_length, park_frequency, color=plotstyle.BLACK, lw=0.9, ls=(0, (3.0, 1.6)),
              zorder=4)
    ax_c.set(xlim=(1.8, 2.8), ylim=(1.0, 2.4), xticks=np.arange(1.8, 2.81, 0.2),
             yticks=[1.0, 1.4, 1.8, 2.2], xlabel="Deployed length [m]", ylabel="Frequency [Hz]")
    ax_c.text(2.2, 1.80, f"Max. diff. {100.0 * beam_frequency_error.max():.2f}%",
              ha="left", va="top", fontsize=7,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")):
        ax.grid(color="0.88", lw=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(width=0.6, length=2.4)
        ax.yaxis.set_label_coords(-8.5 / (W * (right - left)), 0.5)
        plotstyle.panel_label(fig, 0.5 / W, ax.get_position().y1 + 1.0 / H, letter)
    plotstyle.save(fig, paths.figure_path("fig3_ANCF_ALE_validation"), show=render)


def main():
    args = parser(__doc__, figures_only=True).parse_args()
    if not args.figures_only:
        run_jobs(simulate_case, [dict(speed_name=s, model=m, force=args.force, render=args.render)
                                 for s in CASES for m in ("beam", "ale")],
                 1 if args.render else args.workers)
    plotstyle.apply(interactive=args.render)
    evaluate_and_plot(render=args.render)


if __name__ == "__main__":
    main()
