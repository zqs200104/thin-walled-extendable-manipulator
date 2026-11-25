"""Cross-section of the thin-walled boom: a half-open semi-annulus (Section 2.4).

The section coordinates are (eta, zeta) with eta along the symmetry axis. The
circle centre lies at eta = 0 and the wall occupies eta < 0. The ANCF nodes are
placed on the centroidal axis, so all sectional properties refer to the
centroid.
"""
import numpy as np
import exudyn as exu

from . import params as P


def section_properties(r_outer=P.R_OUTER, r_inner=P.R_INNER, e_mod=P.E_EQ,
                       nu=P.NU_EQ, rho=P.RHO_EQ, k_eta=P.K_ETA, k_zeta=P.K_ZETA):
    """Return a dict with the geometric, stiffness and inertia properties."""
    t_wall = r_outer - r_inner
    r_mid = 0.5 * (r_outer + r_inner)
    area = 0.5 * np.pi * (r_outer**2 - r_inner**2)
    # Centroid on the symmetry axis (negative: the wall lies at eta < 0).
    eta_c = -(4.0 / (3.0 * np.pi)) * (r_outer**3 - r_inner**3) / (r_outer**2 - r_inner**2)
    i_circle = (np.pi / 8.0) * (r_outer**4 - r_inner**4)     # about the circle centre
    # Exudyn's beam frame is (x, y, z) = (xi, eta, zeta). The section is
    # symmetric about eta, so the second moment about eta (I_y) is unchanged by
    # the shift to the centroid, while the one about zeta (I_z) loses A*eta_c^2.
    I_y = i_circle
    I_z = i_circle - area * eta_c**2
    j_torsion = np.pi * r_mid * t_wall**3 / 3.0              # open thin-walled section
    g_mod = e_mod / (2.0 * (1.0 + nu))
    return {
        "A": area, "eta_c": eta_c, "r_mid": r_mid,
        "I_y": I_y, "I_z": I_z, "J_t": j_torsion, "J_mass": I_y + I_z,
        "E": e_mod, "G": g_mod, "rho": rho,
        "EA": e_mod * area, "GA_eta": g_mod * area * k_eta, "GA_zeta": g_mod * area * k_zeta,
        "GJ": g_mod * j_torsion, "EI_y": e_mod * I_y, "EI_z": e_mod * I_z,
        "rhoA": rho * area,
        # offset d_c from the wall midline (URDF point of joint 3) to the centroid
        "d_c": eta_c + r_mid,
    }


def beam_section(props):
    """Exudyn BeamSection with the diagonal stiffness of Eq. (26).

    ObjectANCFBeam takes only the diagonal of the stiffness matrix, the
    rotary inertia and the mass per length; a damping matrix would be ignored,
    so the boom has no structural damping and the root damper is the only
    dissipation of the model.
    """
    section = exu.BeamSection()
    section.stiffnessMatrix = np.diag([
        props["EA"], props["GA_eta"], props["GA_zeta"],
        props["GJ"], props["EI_y"], props["EI_z"],
    ])
    section.inertia = np.diag([props["rho"] * props["J_mass"],
                               props["rho"] * props["I_y"],
                               props["rho"] * props["I_z"]])
    section.massPerLength = props["rhoA"]
    return section


def section_geometry(props, r_outer=P.R_OUTER, r_inner=P.R_INNER, n_arc=180):
    """Polygon of the section relative to the centroid, used for drawing only."""
    geometry = exu.BeamSectionGeometry()
    polygon = exu.Vector2DList()
    for i in range(n_arc + 1):
        th = 0.5 * np.pi + i * np.pi / n_arc
        polygon.Append([r_outer * np.cos(th) - props["eta_c"], r_outer * np.sin(th)])
    for i in range(n_arc + 1):
        th = 1.5 * np.pi - i * np.pi / n_arc
        polygon.Append([r_inner * np.cos(th) - props["eta_c"], r_inner * np.sin(th)])
    geometry.polygonalPoints = polygon
    return geometry
