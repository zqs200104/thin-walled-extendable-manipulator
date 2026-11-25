"""Frozen-configuration eigenanalysis of the manipulator (frozen-pose modes of Section 4.6 and s11).

The model is assembled at fixed joint angles and deployed length, with the
joints locked by their coordinate constraints. Because the slope coordinates
of the ANCF nodes and the constrained coordinates leave the mass matrix
singular, the generalized problem K v = w^2 M v is solved on the null space of
the constraint Jacobian.

The stiffness matrix comes from the numerical differentiation of the solver. Its
default step (relativeEpsilon 1e-7 times minimumCoordinateSize 0.01) gives a
forward difference error that shifts the low root-spring mode by up to 1.5 %,
because the axial stiffness of the boom is about 5e8 N/m. The much smaller step
FD_RELATIVE_EPSILON removes this error; the time integration is not affected.
"""
import numpy as np
import exudyn as exu
from scipy.linalg import eig, svd

from .model import Manipulator

FD_RELATIVE_EPSILON = 1.0e-9


class _Hold:
    """Command that keeps the joints and the length at their initial values."""

    def __init__(self, q, length):
        self.q, self.length = np.asarray(q, dtype=float), float(length)

    def joint_pva(self, t):
        return self.q, np.zeros(2), np.zeros(2)

    def length_pva(self, t):
        return self.length, 0.0, 0.0


def frozen_modes(q, length, n_modes=24, f_min=0.05):
    """Natural frequencies [Hz] and the tip displacement of every mode in the boom frame.

    Returns rows (frequency, tip_xi, tip_eta, tip_zeta) of the mass-normalised
    modes, so the participation of a mode at the tip can be compared within
    one configuration.
    """
    sc = exu.SystemContainer()
    mbs = sc.AddSystem()
    model = Manipulator(mbs, _Hold(q, length), q, length, gravity=False, joint_drive="constraint")
    mbs.Assemble()
    settings = exu.SimulationSettings()
    settings.linearSolverType = exu.LinearSolverType.EigenSparse
    settings.solutionSettings.writeSolutionToFile = False
    settings.staticSolver.verboseMode = 0
    settings.staticSolver.newton.numericalDifferentiation.relativeEpsilon = FD_RELATIVE_EPSILON
    solver = exu.MainSolverStatic()
    solver.InitializeSolver(mbs, settings)
    try:
        n2, n1, nae = solver.GetODE2size(), solver.GetODE1size(), solver.GetAEsize()
        solver.ComputeMassMatrix(mbs)
        mass = np.asarray(solver.GetSystemMassMatrix(), dtype=float)
        solver.ComputeJacobianODE2RHS(mbs, scalarFactor_ODE2=-1.0, scalarFactor_ODE2_t=0.0, scalarFactor_ODE1=0.0)
        solver.ComputeJacobianAE(mbs, scalarFactor_ODE2=1.0, scalarFactor_ODE2_t=0.0, scalarFactor_ODE1=0.0,
                                 velocityLevel=False)
        jacobian = np.asarray(solver.GetSystemJacobian(), dtype=float)
    finally:
        solver.FinalizeSolver(mbs, settings)
    stiffness = jacobian[:n2, :n2]
    constraint = jacobian[:n2, n2 + n1:n2 + n1 + nae]
    left, singular, _ = svd(constraint, full_matrices=True)
    tol = max(constraint.shape) * np.finfo(float).eps * max(float(singular[0]), 1.0)
    null = left[:, int(np.sum(singular > tol)):].T
    k_red, m_red = null @ stiffness @ null.T, null @ mass @ null.T
    values, vectors = eig(k_red, m_red, check_finite=False)
    ok = (np.isfinite(values.real) & np.isfinite(values.imag) & (values.real > 0.0)
          & (np.abs(values.imag) <= 1.0e-6 * np.maximum(values.real, 1.0)))
    values, vectors = values.real[ok], np.real(vectors[:, ok])
    order = np.argsort(values)[:n_modes]
    values, vectors = values[order], vectors[:, order]
    modes = null.T @ vectors
    for j in range(vectors.shape[1]):                      # mass normalisation
        m_j = float(vectors[:, j] @ m_red @ vectors[:, j])
        if m_j > 1.0e-18:
            modes[:, j] /= np.sqrt(m_j)
    frequencies = np.sqrt(values) / (2.0 * np.pi)

    # tip displacement of each mode, by a small perturbation of the coordinates
    base = np.asarray(mbs.systemData.GetODE2Coordinates(), dtype=float)
    out_type, cfg = exu.OutputVariableType.Position, exu.ConfigurationType.Current
    tip0 = np.asarray(mbs.GetMarkerOutput(model.tip_marker, out_type, cfg), dtype=float)
    axes = np.column_stack(model.axes)
    rows = []
    try:
        for j, f in enumerate(frequencies):
            mbs.systemData.SetODE2Coordinates((base + 1.0e-6 * modes[:, j]).tolist(), cfg)
            tip = (np.asarray(mbs.GetMarkerOutput(model.tip_marker, out_type, cfg), dtype=float) - tip0) / 1.0e-6
            rows.append([f, *(axes.T @ tip)])
    finally:
        mbs.systemData.SetODE2Coordinates(base.tolist(), cfg)
    rows = np.asarray(rows)
    return rows[rows[:, 0] >= f_min]


def damped_modes(q, length, gravity=False, n_modes=2, f_max=150.0):
    """Natural frequency, damped frequency [Hz] and damping ratio of the lowest modes.

    Solves (lam^2 M + lam D + K) v = 0 on the null space of the constraint
    Jacobian, because the mass matrix is singular (see the module docstring).
    D holds the root damper. With gravity the matrices are taken about the
    static equilibrium; the stiffness then leaves out the terms of the
    constraint reactions.
    """
    sc = exu.SystemContainer()
    mbs = sc.AddSystem()
    Manipulator(mbs, _Hold(q, length), q, length, gravity=gravity, joint_drive="constraint")
    mbs.Assemble()
    settings = exu.SimulationSettings()
    settings.linearSolverType = exu.LinearSolverType.EigenSparse
    settings.solutionSettings.writeSolutionToFile = False
    settings.staticSolver.verboseMode = 0
    settings.staticSolver.newton.numericalDifferentiation.relativeEpsilon = FD_RELATIVE_EPSILON
    if gravity:
        settings.staticSolver.numberOfLoadSteps = 10
        settings.staticSolver.newton.relativeTolerance = 1.0e-8
        settings.staticSolver.newton.absoluteTolerance = 1.0e-6
        mbs.SolveStatic(settings)
        equilibrium = mbs.systemData.GetODE2Coordinates()
        mbs.systemData.SetODE2Coordinates(equilibrium, exu.ConfigurationType.Initial)
    solver = exu.MainSolverStatic()
    solver.InitializeSolver(mbs, settings)
    try:
        n2, n1, nae = solver.GetODE2size(), solver.GetODE1size(), solver.GetAEsize()
        solver.ComputeMassMatrix(mbs)
        mass = np.asarray(solver.GetSystemMassMatrix(), dtype=float).copy()
        solver.ComputeJacobianODE2RHS(mbs, scalarFactor_ODE2=-1.0, scalarFactor_ODE2_t=0.0, scalarFactor_ODE1=0.0)
        solver.ComputeJacobianAE(mbs, scalarFactor_ODE2=1.0, scalarFactor_ODE2_t=0.0, scalarFactor_ODE1=0.0,
                                 velocityLevel=False)
        jacobian = np.asarray(solver.GetSystemJacobian(), dtype=float).copy()
        solver.ComputeJacobianODE2RHS(mbs, scalarFactor_ODE2=0.0, scalarFactor_ODE2_t=-1.0, scalarFactor_ODE1=0.0)
        damping = np.asarray(solver.GetSystemJacobian(), dtype=float)[:n2, :n2].copy()
    finally:
        solver.FinalizeSolver(mbs, settings)
    stiffness = jacobian[:n2, :n2]
    constraint = jacobian[:n2, n2 + n1:n2 + n1 + nae]
    left, singular, _ = svd(constraint, full_matrices=True)
    tol = max(constraint.shape) * np.finfo(float).eps * max(float(singular[0]), 1.0)
    null = left[:, int(np.sum(singular > tol)):]
    m_red, k_red, d_red = null.T @ mass @ null, null.T @ stiffness @ null, null.T @ damping @ null
    n = m_red.shape[0]
    a = np.block([[np.zeros((n, n)), np.eye(n)], [-k_red, -d_red]])
    b = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), m_red]])
    values = eig(a, b, right=False, check_finite=False)
    values = values[np.isfinite(values)]
    values = values[(values.imag > 1.0) & (np.abs(values) < 2.0 * np.pi * f_max)]
    values = values[np.argsort(values.imag)][:n_modes]
    return [(float(abs(v) / (2.0 * np.pi)), float(v.imag / (2.0 * np.pi)), float(-v.real / abs(v))) for v in values]


def dominant_planar_mode(rows, f_band=(0.10, 2.00)):
    """Mode with the largest tip motion along eta (in the path plane) within f_band."""
    keep = (rows[:, 0] >= f_band[0]) & (rows[:, 0] <= f_band[1])
    candidates = np.flatnonzero(keep)
    return float(rows[candidates[np.argmax(np.abs(rows[candidates, 2]))], 0])
