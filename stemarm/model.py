"""Flexible multibody model of the manipulator (Section 2) in Exudyn.

Structure of the model:
  * joints R1 and R2: Exudyn kinematic tree built from the URDF (arm.py);
  * boom: N ANCF beam elements (ObjectANCFBeam) on the centroidal axis;
  * root connection: generic joint that leaves only the rotation about zeta
    free, plus a torsional spring-damper (K, D) about that axis, Eq. (33);
  * gripper: rigid body fixed to the tip node.

Joint motion is prescribed. The deployed length is changed by updating the
reference length of every element before each time step (Section 4.1).
"""
import numpy as np
import exudyn as exu
import exudyn.graphics as graphics
from exudyn.itemInterface import (
    NodePointSlope23, ObjectANCFBeam, VANCFBeam, MarkerNodePosition, MarkerNodeRigid,
    MarkerKinematicTreeRigid, MarkerBodyRigid, MarkerNodeCoordinate, LoadForceVector,
    GenericJoint, VGenericJoint, TorsionalSpringDamper, VTorsionalSpringDamper,
    NodePointGround, CoordinateConstraint, VCoordinateConstraint, SensorNode, SensorMarker,
    SensorObject,
)
from exudyn.rigidBodyUtilities import RigidBodyInertia

from . import params as P
from .arm import read_urdf, build_robot, rpy_to_rot
from .section import section_properties, beam_section, section_geometry

PARENT_LINK = 1          # index of link 2 (second_link) in the kinematic tree


def normalize(v):
    return v / np.linalg.norm(v)


def rot_axis_angle(axis, theta):
    """Rodrigues rotation about a unit axis."""
    a = normalize(np.asarray(axis, dtype=float))
    skew = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


# Beam axes in frame {3}: xi = z3, eta = x3, zeta = y3 (Section 2.2)
EX_3 = np.array([0.0, 0.0, 1.0])
EY_3 = np.array([1.0, 0.0, 0.0])
EZ_3 = np.cross(EX_3, EY_3)


def mount_rotation(bias_deg=P.BIAS_DEG):
    """Rotation of frame {r} relative to link 2: frame {3} turned by beta about zeta."""
    return rpy_to_rot(P.JOINT3_RPY) @ rot_axis_angle(EZ_3, np.deg2rad(bias_deg))


def beam_frame(r_parent, r_mount):
    """Unit vectors (xi, eta, zeta) of the undeformed boom in the global frame."""
    r = r_parent @ r_mount
    ex = normalize(r[:, 2])
    ey = r[:, 0] - np.dot(r[:, 0], ex) * ex
    ey = normalize(ey)
    ez = normalize(np.cross(ex, ey))
    return ex, ey, ez


def root_point_in_link2(props, r_mount):
    """Root node position in the frame of link 2: joint-3 point + d_a xi + d_c eta."""
    return P.JOINT3_XYZ + r_mount @ (P.ROOT_AXIAL_OFFSET * EX_3 + props["d_c"] * EY_3)


def marker_offset(robot, props, r_mount, q_cal=P.MARKER_CALIBRATION_Q_DEG):
    """Offset of marker point P from the tip node, in the tip-node frame [m].

    Both measured points (marker and inner boom surface, phi = 180 deg) were
    taken at L = 0, q1 = 0 deg, q2 = 90 deg.
    """
    ht = np.asarray(robot.LinkHT(np.deg2rad(q_cal))[PARENT_LINK], dtype=float)
    ex, ey, ez = beam_frame(ht[0:3, 0:3], r_mount)
    r_cal = np.column_stack((ex, ey, ez))
    delta_local = r_cal.T @ (P.MARKER_POINT_MM - P.ROD_POINT_MM)
    phi = np.deg2rad(P.ROD_POINT_PHI_DEG)
    rod_local = np.array([1e3 * P.R_INNER * np.cos(phi), 1e3 * P.R_INNER * np.sin(phi)])
    rod_to_node = np.array([0.0, 1e3 * props["eta_c"] - rod_local[0], -rod_local[1]])
    return 1.0e-3 * (delta_local - rod_to_node)


def gripper_inertia():
    """Gripper inertia about the tip node, expressed in the boom frame."""
    r = rpy_to_rot(P.GRIPPER_RPY)
    com = P.GRIPPER_XYZ + r @ P.GRIPPER_COM
    inertia = r @ P.GRIPPER_INERTIA @ r.T
    inertia = 0.5 * (inertia + inertia.T)
    return RigidBodyInertia(mass=P.GRIPPER_MASS, inertiaTensor=inertia).Translated(com.tolist())


class Manipulator:
    """Builds the model into an Exudyn system and keeps the item numbers.

    Parameters
    ----------
    mbs : exudyn MainSystem
    command : object with ``joint_pva(t)`` -> (q, qd, qdd) arrays of length 2 and
        ``length_pva(t)`` -> (L, Ld, Ldd); see commands.py
    q0 : initial joint angles [rad]
    length0 : deployed length at t = 0 [m]
    gravity : include gravity on links, boom and gripper
    joint_drive : 'constraint' prescribes q(t) through coordinate constraints;
        'state' overwrites the joint coordinates before each step (used for the
        fixed-length ground slews of Sections 5.2 and 5.3, see USER_MANUAL.md).
    variable_length : update the element length from command.length_pva(t)
    """

    def __init__(self, mbs, command, q0, length0, gravity=True, joint_drive="constraint",
                 variable_length=False, n_elements=P.N_ELEMENTS, with_graphics=False):
        self.mbs, self.command = mbs, command
        self.n_elements = n_elements
        self.variable_length = variable_length
        self.gravity = np.asarray(P.GRAVITY if gravity else np.zeros(3), dtype=float)
        self.props = section_properties()

        mbs.CreateGround(graphicsDataList=[graphics.CheckerBoard(point=[0, 0, 0], size=max(3.0, 2.5 * length0))]
                         if with_graphics else [])

        # ---- rigid joint chain (R1, R2) ------------------------------------
        arm = read_urdf()
        self.joint_limits = [link["limits"] for link in arm["chain"]]
        self.robot = build_robot(arm, self.gravity, with_graphics)
        tree = self.robot.CreateKinematicTree(mbs)
        self.oKT, self.nKT = tree["objectKinematicTree"], tree["nodeGeneric"]
        q0 = np.asarray(q0, dtype=float)
        mbs.SetNodeParameter(self.nKT, "initialCoordinates", q0)
        for name in ("jointForceVector", "jointPositionOffsetVector", "jointVelocityOffsetVector"):
            mbs.SetObjectParameter(self.oKT, name, [0.0] * len(q0))   # no internal joint control
        self.sensors = {
            "q": mbs.AddSensor(SensorNode(nodeNumber=self.nKT, storeInternal=True, writeToFile=False,
                                          outputVariableType=exu.OutputVariableType.Coordinates)),
            "qd": mbs.AddSensor(SensorNode(nodeNumber=self.nKT, storeInternal=True, writeToFile=False,
                                           outputVariableType=exu.OutputVariableType.Coordinates_t)),
        }
        if joint_drive == "constraint":
            self._add_joint_constraints(q0)
        elif joint_drive != "state":
            raise ValueError("joint_drive must be 'constraint' or 'state'")
        self.joint_drive = joint_drive

        # ---- boom geometry at the initial pose ------------------------------
        ht = np.asarray(self.robot.LinkHT(q0)[PARENT_LINK], dtype=float)
        p_parent, r_parent = ht[0:3, 3], ht[0:3, 0:3]
        self.r_mount = mount_rotation()
        ex, ey, ez = beam_frame(r_parent, self.r_mount)
        self.axes = (ex, ey, ez)            # undeformed boom axes xi, eta, zeta
        p_three = p_parent + r_parent @ P.JOINT3_XYZ
        p_root = p_three + P.ROOT_AXIAL_OFFSET * ex + self.props["d_c"] * ey
        self.root_local = root_point_in_link2(self.props, self.r_mount)
        self.marker_offset = marker_offset(self.robot, self.props, self.r_mount)

        # ---- ANCF boom -----------------------------------------------------
        section = beam_section(self.props)
        geometry = section_geometry(self.props)
        l_elem = length0 / n_elements
        n0 = mbs.AddNode(NodePointSlope23(referenceCoordinates=list(p_root) + list(ey) + list(ez)))
        self.first_node, nodes, self.elements = n0, [n0], []
        colors = ([0.10, 0.25, 0.65, 1.0], [0.70, 0.80, 0.95, 1.0])   # alternate shades show the mesh
        for k in range(n_elements):
            n1 = mbs.AddNode(NodePointSlope23(
                referenceCoordinates=list(p_root + (k + 1) * l_elem * ex) + list(ey) + list(ez)))
            self.elements.append(mbs.AddObject(ObjectANCFBeam(
                nodeNumbers=[n0, n1], physicsLength=l_elem, sectionData=section,
                crossSectionPenaltyFactor=P.CROSS_SECTION_PENALTY,
                visualization=VANCFBeam(sectionGeometry=geometry, color=colors[k % 2]))))
            nodes.append(n1)
            n0 = n1
        self.last_node = n0

        # Self-weight as nodal loads (ObjectANCFBeam has no body-mass marker):
        # half an element mass at the end nodes, one element mass inside.
        self.gravity_loads, self.gravity_weights = [], []
        if gravity:
            m_elem = self.props["rhoA"] * l_elem
            for i, node in enumerate(nodes):
                weight = 0.5 if i in (0, len(nodes) - 1) else 1.0
                marker = mbs.AddMarker(MarkerNodePosition(nodeNumber=node))
                self.gravity_loads.append(mbs.AddLoad(LoadForceVector(
                    markerNumber=marker, loadVector=(weight * m_elem * self.gravity).tolist())))
                self.gravity_weights.append(weight)

        # ---- root connection: generic joint (rotation about zeta free) + TSD
        m_root_link = mbs.AddMarker(MarkerKinematicTreeRigid(
            objectNumber=self.oKT, linkNumber=PARENT_LINK, localPosition=self.root_local.tolist()))
        frame_in_link2 = np.column_stack((self.r_mount @ EX_3, self.r_mount @ EY_3, self.r_mount @ EZ_3))
        m_root_node = mbs.AddMarker(MarkerNodeRigid(nodeNumber=self.first_node))
        mbs.AddObject(GenericJoint(
            markerNumbers=[m_root_link, m_root_node], constrainedAxes=[1, 1, 1, 1, 1, 0],
            rotationMarker0=frame_in_link2, rotationMarker1=np.eye(3),
            visualization=VGenericJoint(show=True, axesRadius=0.003, axesLength=0.04)))
        o_tsd = mbs.AddObject(TorsionalSpringDamper(
            markerNumbers=[m_root_link, m_root_node], stiffness=P.K_ROOT, damping=P.D_ROOT,
            rotationMarker0=frame_in_link2, rotationMarker1=np.eye(3),
            offset=0.0, velocityOffset=0.0, torque=0.0,
            visualization=VTorsionalSpringDamper(show=True, drawSize=0.02)))
        self.sensors["theta_root"] = mbs.AddSensor(SensorObject(
            objectNumber=o_tsd, storeInternal=True, writeToFile=False,
            outputVariableType=exu.OutputVariableType.Rotation))

        m_tip_pos = mbs.AddMarker(MarkerNodePosition(nodeNumber=self.last_node))
        self.tip_marker = m_tip_pos
        self.sensors["tip_pos"] = mbs.AddSensor(SensorMarker(
            markerNumber=m_tip_pos, storeInternal=True, writeToFile=False,
            outputVariableType=exu.OutputVariableType.Position))
        m_tip_rigid = mbs.AddMarker(MarkerNodeRigid(nodeNumber=self.last_node))
        self.sensors["tip_rot"] = mbs.AddSensor(SensorMarker(
            markerNumber=m_tip_rigid, storeInternal=True, writeToFile=False,
            outputVariableType=exu.OutputVariableType.RotationMatrix))
        m_root_pos = mbs.AddMarker(MarkerNodePosition(nodeNumber=self.first_node))
        self.sensors["root_pos"] = mbs.AddSensor(SensorMarker(
            markerNumber=m_root_pos, storeInternal=True, writeToFile=False,
            outputVariableType=exu.OutputVariableType.Position))
        self.sensors["root_rot"] = mbs.AddSensor(SensorMarker(
            markerNumber=m_root_node, storeInternal=True, writeToFile=False,
            outputVariableType=exu.OutputVariableType.RotationMatrix))

        # ---- gripper: rigid body fixed to the tip node ----------------------
        gripper_graphics = []
        if with_graphics:
            from .arm import MESH_DIR
            import os
            stl = graphics.FromSTLfile(os.path.join(MESH_DIR, "four_link.STL"),
                                       color=graphics.color.steelblue, verbose=False)
            gripper_graphics = [graphics.Move(stl, P.GRIPPER_XYZ.tolist(), rpy_to_rot(P.GRIPPER_RPY))]
        body_kwargs = dict(referencePosition=(p_root + length0 * ex).tolist(),
                           referenceRotationMatrix=np.column_stack((ex, ey, ez)),
                           inertia=gripper_inertia(), graphicsDataList=gripper_graphics)
        if gravity:
            body_kwargs["gravity"] = self.gravity.tolist()
        o_gripper = mbs.CreateRigidBody(**body_kwargs)
        m_gripper = mbs.AddMarker(MarkerBodyRigid(bodyNumber=o_gripper, localPosition=[0, 0, 0]))
        mbs.AddObject(GenericJoint(markerNumbers=[m_tip_rigid, m_gripper],
                                   constrainedAxes=[1, 1, 1, 1, 1, 1],
                                   visualization=VGenericJoint(show=False)))

        self.length_history = []
        self._joint_slice = None
        mbs.SetPreStepUserFunction(self._pre_step)

    # ------------------------------------------------------------------------
    def _add_joint_constraints(self, q0):
        """q_j(t) = command(t) imposed in every Newton iteration."""
        mbs = self.mbs
        n_ground = mbs.AddNode(NodePointGround(referenceCoordinates=[0.0, 0.0, 0.0]))
        m_ground = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n_ground, coordinate=0))
        for j in range(len(q0)):
            m_joint = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=self.nKT, coordinate=j))
            mbs.AddObject(CoordinateConstraint(
                markerNumbers=[m_ground, m_joint], offset=float(q0[j]),
                offsetUserFunction=lambda mbs, t, item, offset, j=j: float(self.command.joint_pva(t)[0][j]),
                offsetUserFunction_t=lambda mbs, t, item, offset, j=j: float(self.command.joint_pva(t)[1][j]),
                visualization=VCoordinateConstraint(show=False)))

    def _pre_step(self, mbs, t):
        if self.variable_length:
            self.set_length(max(0.0, float(t)))
        if self.joint_drive == "state":
            self._overwrite_joint_state(t)
        return True

    def set_length(self, t):
        """Element length l_e = L(t)/N, Eq. (13); the nodal weights follow L(t)."""
        length, velocity, _ = self.command.length_pva(t)
        l_elem = length / float(self.n_elements)
        for element in self.elements:
            self.mbs.SetObjectParameter(element, "physicsLength", float(l_elem))
        m_elem = float(self.props["rhoA"] * l_elem)
        for load, weight in zip(self.gravity_loads, self.gravity_weights):
            self.mbs.SetLoadParameter(load, "loadVector", (float(weight) * m_elem * self.gravity).tolist())
        self.length_history.append((t, length, velocity))

    def _overwrite_joint_state(self, t):
        q, qd, qdd = self.command.joint_pva(t)
        system = self.mbs.systemData
        if self._joint_slice is None:
            start = self.mbs.GetNodeODE2Index(self.nKT)
            self._joint_slice = slice(start, start + len(q))
        u = np.array(system.GetODE2Coordinates(), dtype=float)
        v = np.array(system.GetODE2Coordinates_t(), dtype=float)
        a = np.array(system.GetODE2Coordinates_tt(), dtype=float)
        u[self._joint_slice], v[self._joint_slice], a[self._joint_slice] = q, qd, qdd
        for cfg in (exu.ConfigurationType.Current, exu.ConfigurationType.Visualization):
            system.SetODE2Coordinates(u, configuration=cfg)
            system.SetODE2Coordinates_t(v, configuration=cfg)
            system.SetODE2Coordinates_tt(a, configuration=cfg)

    # ------------------------------------------------------------------------
    def rigid_tip(self, q, length):
        """FK(q) of Eq. (46): tip of the undeformed boom on its centroidal axis."""
        return rigid_tip(self.robot, self.root_local, self.r_mount, q, length)


def rigid_tip(robot, root_local, r_mount, q, length):
    """FK(q) of Eq. (46): tip of the undeformed boom on its centroidal axis."""
    ht = np.asarray(robot.LinkHT(np.asarray(q, dtype=float))[PARENT_LINK], dtype=float)
    r = ht[0:3, 0:3]
    axis = normalize(r @ normalize(r_mount @ EX_3))
    return ht[0:3, 3] + r @ root_local + float(length) * axis


def kinematics():
    """Robot, root point and mount rotation for kinematic calculations without a model."""
    arm = read_urdf()
    r_mount = mount_rotation()
    return build_robot(arm, np.zeros(3)), root_point_in_link2(section_properties(), r_mount), r_mount, arm
