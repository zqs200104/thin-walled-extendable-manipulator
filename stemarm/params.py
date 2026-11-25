"""Model parameters of the paper (Table 2).

All values are in SI units unless the name says otherwise. Every script in
this repository reads its parameters from here, so a change in one place
changes all simulations.
"""
import numpy as np

# ----------------------------------------------------------------------------
# Thin-walled boom (semi-annular open section), Table 2
# ----------------------------------------------------------------------------
R_OUTER = 25.50e-3          # outer radius R_o [m]
R_INNER = 25.00e-3          # inner radius R_i [m]
E_EQ = 130e9                # equivalent Young's modulus [Pa]
NU_EQ = 0.3                 # equivalent Poisson's ratio
RHO_EQ = 1800.0             # equivalent density [kg/m^3]
K_ETA = 0.39                # shear correction factor along the symmetry axis (eta)
K_ZETA = 0.50               # shear correction factor normal to it (zeta)
N_ELEMENTS = 48             # number of ANCF elements of the manipulator boom
# Exudyn adds a cross-section deformation penalty to ObjectANCFBeam; this is
# the factor used in all simulations of the paper.
CROSS_SECTION_PENALTY = [0.05, 0.05, 0.05]

# ----------------------------------------------------------------------------
# Root connection (Section 2.5), identified in Section 4.3
# ----------------------------------------------------------------------------
K_ROOT = 64.83323855953904  # rotational stiffness K_theta_zeta [N m/rad]
D_ROOT = 0.15411            # rotational damping D_theta_zeta [N m s/rad]
BIAS_DEG = 5.785            # mounting bias beta about the section axis zeta [deg]
ROOT_AXIAL_OFFSET = 1.10e-3  # axial offset d_a of the root node along xi [m]

# ----------------------------------------------------------------------------
# Constant transforms (Table 1). Frame {3} is the boom mount on link 2,
# given in the frame of link 2. Its x, y, z axes are eta, zeta, xi.
# ----------------------------------------------------------------------------
JOINT3_XYZ = np.array([-0.049694, 0.185, -0.0020004])
JOINT3_RPY = np.array([-1.5708, 0.0, 0.0])
# The URDF point of joint 3 lies on the wall midline (radius (R_o+R_i)/2); the
# ANCF nodes lie on the centroidal axis. The offset d_c between both points is
# computed from the section (see section.py).

# Gripper (link 4 of the CAD model): mass, centre of mass and inertia in the
# gripper frame, and the pose of that frame relative to the boom tip node.
GRIPPER_MASS = 0.125307061960107
GRIPPER_COM = np.array([0.0112305683665844, 0.000998229590755776, -0.0629627569952952])
GRIPPER_INERTIA = np.array([
    [5.14848383004205e-05, -1.32288165438899e-07, 6.18656482955177e-07],
    [-1.32288165438899e-07, 2.81870502598857e-05, -1.72477632702786e-07],
    [6.18656482955177e-07, -1.72477632702786e-07, 3.82649153748310e-05],
])
GRIPPER_XYZ = np.array([0.092809, 0.0016939, -0.0020004])   # in the beam frame (xi, eta, zeta)
GRIPPER_RPY = np.array([0.5 * np.pi, 0.0, 0.5 * np.pi])

# ----------------------------------------------------------------------------
# Marker point P of the motion-capture system (Section 4.1). Two points were
# measured at L = 0, q1 = 0 deg, q2 = 90 deg: the marker itself and a point on
# the inner boom surface (phi = 180 deg, i.e. (eta, zeta) = (-R_i, 0)).
# ----------------------------------------------------------------------------
MARKER_POINT_MM = np.array([-112.75, 129.86, 219.73])
ROD_POINT_MM = np.array([-69.08, 129.10, 167.183])
ROD_POINT_PHI_DEG = 180.0
MARKER_CALIBRATION_Q_DEG = (0.0, 90.0)
# Shift of P along the boom axis for the slews of 2026-11-13 (Sections 5.2-5.3),
# set from the final X position after Case A (first unshaped trial); used only by s06 [mm].
MARKER_AXIAL_SHIFT_SLEW_MM = -25.79
# Motion-capture origin expressed in the simulation frame [mm].
MOCAP_ORIGIN_OFFSET_MM = np.array([128.28, 128.28, 0.0])

# ----------------------------------------------------------------------------
# Joint trajectories of the ground slews (Section 5.2)
# ----------------------------------------------------------------------------
JOINT_VMAX = 1.0            # [rad/s]
JOINT_AMAX = 2.0            # [rad/s^2]
# The host PC samples commands at 5 ms; the synchronised duration is never
# shorter than one sample (this only matters for very short moves).
HOST_TRAJ_DT = 0.005

# Ground slews of Cases A and B: initial and final joint angles [deg]
CASES = {
    "A": {"q_start_deg": (0.0, 10.0), "q_end_deg": (0.0, 90.0)},
    "B": {"q_start_deg": (0.0, 10.0), "q_end_deg": (90.0, 90.0)},
}

# ----------------------------------------------------------------------------
# Time integration (Section 4.1): generalized-alpha, h = 1 ms, rho_inf = 0.5
# ----------------------------------------------------------------------------
STEP_SIZE = 1.0e-3
SPECTRAL_RADIUS = 0.5
SENSOR_PERIOD = 2.0e-3      # sampling of the stored responses (500 Hz)
GRAVITY = np.array([0.0, 0.0, -9.81])
