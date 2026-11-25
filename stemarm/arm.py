"""Rigid joint chain of the manipulator (joints R1 and R2) read from the URDF.

Only the two revolute joints are simulated as rigid links. The boom replaces
joint T1 and is modelled with ANCF elements (see model.py); the gripper is
attached to the boom tip. The URDF is parsed directly, so no ROS or robotics
toolbox is needed.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from .paths import MODEL_DIR

URDF_FILE = os.path.join(MODEL_DIR, "stemarm", "urdf", "stemarm.urdf")
MESH_DIR = os.path.join(MODEL_DIR, "stemarm", "meshes")


def rpy_to_rot(rpy):
    """Rotation matrix of URDF roll-pitch-yaw angles, R = Rz(y) Ry(p) Rx(r)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _vec(text, default="0 0 0"):
    return np.array([float(v) for v in (text or default).split()], dtype=float)


def read_urdf(path=URDF_FILE):
    """Return the movable links in tree order with their joint and inertia data."""
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        mesh = link.find("visual/geometry/mesh")
        color = link.find("visual/material/color")
        entry = {"name": link.get("name"), "mesh": None, "color": None}
        if mesh is not None:
            entry["mesh"] = os.path.join(MESH_DIR, os.path.basename(mesh.get("filename")))
        if color is not None:
            entry["color"] = [float(v) for v in color.get("rgba").split()]
        if inertial is not None:
            origin = inertial.find("origin")
            inertia = inertial.find("inertia").attrib
            ixx, ixy, ixz = (float(inertia[k]) for k in ("ixx", "ixy", "ixz"))
            iyy, iyz, izz = (float(inertia[k]) for k in ("iyy", "iyz", "izz"))
            entry.update({
                "mass": float(inertial.find("mass").get("value")),
                "com": _vec(origin.get("xyz") if origin is not None else None),
                "inertia": np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]),
            })
        links[entry["name"]] = entry

    chain = []
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            raise ValueError(f"unexpected joint type {joint.get('type')!r}")
        if not np.allclose(_vec(joint.find("axis").get("xyz")), [0.0, 0.0, 1.0]):
            raise ValueError("all joints of the arm rotate about their local +z axis")
        origin = joint.find("origin")
        pre_ht = np.eye(4)
        pre_ht[0:3, 0:3] = rpy_to_rot(_vec(origin.get("rpy")))
        pre_ht[0:3, 3] = _vec(origin.get("xyz"))
        limit = joint.find("limit")
        child = links[joint.find("child").get("link")]
        chain.append({
            **child,
            "joint": joint.get("name"),
            "parent": joint.find("parent").get("link"),
            "preHT": pre_ht,
            "limits": (float(limit.get("lower")), float(limit.get("upper"))),
        })
    base_name = chain[0]["parent"]
    return {"base": links[base_name], "chain": chain}


def build_robot(arm, gravity, with_graphics=False):
    """Exudyn Robot (kinematic tree) of the joint chain, joints driven externally."""
    import exudyn.graphics as graphics
    from exudyn.robotics import Robot, RobotBase, RobotTool, RobotLink, VRobotBase, VRobotTool, VRobotLink
    from exudyn.rigidBodyUtilities import HTtranslate

    def mesh_graphics(link):
        if not with_graphics or link["mesh"] is None:
            return []
        color = link["color"] or graphics.color.grey
        return [graphics.FromSTLfile(link["mesh"], color=color, verbose=False)]

    robot = Robot(
        gravity=list(gravity),
        base=RobotBase(HT=HTtranslate([0, 0, 0]),
                       visualization=VRobotBase(graphicsData=mesh_graphics(arm["base"]))),
        tool=RobotTool(HT=HTtranslate([0, 0, 0]), visualization=VRobotTool(graphicsData=[])),
        referenceConfiguration=[],
    )
    names = [arm["base"]["name"]] + [link["name"] for link in arm["chain"]]
    for link in arm["chain"]:
        robot.AddLink(RobotLink(
            mass=link["mass"],
            parent=names.index(link["parent"]) - 1,   # -1 = base
            COM=link["com"],
            inertia=link["inertia"],
            preHT=link["preHT"],
            jointType="Rz",
            PDcontrol=(0.0, 0.0),
            visualization=VRobotLink(graphicsData=mesh_graphics(link), showMBSjoint=False, showCOM=False),
        ))
    return robot
