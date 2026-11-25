"""Horizontal boom on the root spring, as in the identification tests of Section 4.3.

The arm joints are not needed: in the tests the boom is horizontal
(q1 = 0, q2 = 90 deg), so the root is modelled as fixed to the ground. The
boom axis is tilted by the mounting bias beta in the x-y plane and gravity
acts along -y.
"""
import numpy as np
import exudyn as exu
from exudyn.itemInterface import (
    NodePointSlope23, ObjectANCFBeam, VANCFBeam, MarkerBodyRigid, MarkerNodeRigid, GenericJoint, VGenericJoint,
    TorsionalSpringDamper, VTorsionalSpringDamper, MarkerNodeCoordinate, LoadCoordinate,
)

from . import params as P
from .section import section_properties, beam_section, section_geometry

G = 9.81


def boom_axes(bias_deg=P.BIAS_DEG):
    b = np.deg2rad(bias_deg)
    ex = np.array([np.cos(b), np.sin(b), 0.0])
    ey = np.array([-np.sin(b), np.cos(b), 0.0])
    ex = ex / np.linalg.norm(ex)
    ey = ey - np.dot(ey, ex) * ex
    ey = ey / np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    return ex, ey, ez / np.linalg.norm(ez)


def build(mbs, length, n_elements, k_root, d_root=0.0, beam_gravity=False, props=None, *, with_graphics: bool = False):
    """Boom with N ANCF elements on the root spring-damper; returns a dict of item numbers."""
    props = props or section_properties()
    ground = mbs.CreateGround()
    ex, ey, ez = boom_axes()
    p_root = props["d_c"] * ey                      # nodes on the centroidal axis
    section = beam_section(props)
    visualization = {}
    if with_graphics:
        visualization["visualization"] = VANCFBeam(sectionGeometry=section_geometry(props, n_arc=24),
                                                  color=[0.10, 0.25, 0.65, 1.0])
    l_elem = length / n_elements
    nodes = [mbs.AddNode(NodePointSlope23(referenceCoordinates=list(p_root) + list(ey) + list(ez)))]
    for k in range(n_elements):
        nodes.append(mbs.AddNode(NodePointSlope23(
            referenceCoordinates=list(p_root + (k + 1) * l_elem * ex) + list(ey) + list(ez))))
        mbs.AddObject(ObjectANCFBeam(nodeNumbers=nodes[-2:], physicsLength=l_elem, sectionData=section,
                                     crossSectionPenaltyFactor=P.CROSS_SECTION_PENALTY, **visualization))
    # root: only the rotation about zeta is free, spring-damper about that axis
    ey_r, ez_r = ey / np.linalg.norm(ey), ez / np.linalg.norm(ez)
    ex_r = np.cross(ey_r, ez_r)
    frame = np.column_stack((ex_r / np.linalg.norm(ex_r), ey_r, ez_r))
    m_ground = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ground, localPosition=p_root.tolist()))
    m_root = mbs.AddMarker(MarkerNodeRigid(nodeNumber=nodes[0]))
    mbs.AddObject(GenericJoint(markerNumbers=[m_ground, m_root], constrainedAxes=[1, 1, 1, 1, 1, 0],
                               rotationMarker0=frame, rotationMarker1=np.eye(3), visualization=VGenericJoint(show=False)))
    tsd = mbs.AddObject(TorsionalSpringDamper(markerNumbers=[m_ground, m_root], stiffness=float(k_root),
                                              damping=float(d_root), rotationMarker0=frame, rotationMarker1=np.eye(3),
                                              offset=0.0, velocityOffset=0.0, torque=0.0, activeConnector=True,
                                              visualization=VTorsionalSpringDamper(show=False)))
    if beam_gravity:                                # nodal self-weight
        m_elem = props["rhoA"] * l_elem
        for i, node in enumerate(nodes):
            force = (0.5 if i in (0, len(nodes) - 1) else 1.0) * m_elem * np.array([0.0, -G, 0.0])
            for c in range(3):
                mbs.AddLoad(LoadCoordinate(markerNumber=mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=node,
                                                                                          coordinate=c)),
                                           load=float(force[c])))
    return {"ground": ground, "nodes": nodes, "tip": nodes[-1], "tsd": tsd, "p_root": p_root, "ex": ex}


def static_settings(load_steps=200, adaptive=False, rel_tol=1e-5, abs_tol=1e-8, max_iter=300, stabilizer=1e-7):
    s = exu.SimulationSettings()
    s.displayStatistics = False
    s.linearSolverType = exu.LinearSolverType.EigenSparse
    s.solutionSettings.writeSolutionToFile = False
    st = s.staticSolver
    st.verboseMode = 0
    st.numberOfLoadSteps = load_steps
    st.adaptiveStep = adaptive
    st.newton.relativeTolerance = rel_tol
    st.newton.absoluteTolerance = abs_tol
    st.newton.maxIterations = max_iter
    st.stabilizerODE2term = stabilizer
    return s
