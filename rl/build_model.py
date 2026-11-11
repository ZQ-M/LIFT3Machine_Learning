"""Build the MuJoCo training model (lift3.xml) from the LIFT3 URDF.

What changes relative to the URDF:
  * base_link gets a planar floating base (slide x, slide y, hinge yaw). The four omni wheels are
    welded on as mass only; their traction is applied by the wheel model in lift3_env.py.
  * collision meshes are replaced by boxes / capsules measured from the STL bounding boxes.
  * arm actuators: PD position servos with the assumed DAMIAO torque limits
    (joints 1-3 DM-J4340: 27 N*m peak, joints 4-6 DM-J4310: 7 N*m peak).  NOT confirmed by ARX.
  * lift actuator: the URDF effort (6) cannot hold the ~13.5 kg platform, so a 400 N servo is assumed.
"""
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "..", "LIFT3", "LIFT3", "urdf", "LIFT3.urdf")
OUT = os.path.join(HERE, "lift3.xml")

WHEEL_RADIUS = 0.075
GROUND_Z = WHEEL_RADIUS - 0.03          # base_link origin height above the floor

# assumed arm actuator data, per joint index 1..6
ARM_PEAK_TORQUE = [27.0, 27.0, 27.0, 7.0, 7.0, 7.0]
ARM_KP = [120.0, 120.0, 120.0, 40.0, 40.0, 40.0]
ARM_KD = [6.0, 6.0, 6.0, 2.0, 2.0, 2.0]
ARM_ARMATURE = [0.03, 0.03, 0.03, 0.006, 0.006, 0.006]
ARM_DAMPING = [0.2, 0.2, 0.2, 0.05, 0.05, 0.05]
ARM_FRICTION = [0.15, 0.15, 0.15, 0.05, 0.05, 0.05]
LIFT_FORCE = 400.0

TCP_OFFSET = "0.13 0 0"                  # grasp point between the finger pads, in the gripper link frame

# collision primitives per link: (type, size, pos or fromto)
BODY, LEFT, RIGHT = 4, 1, 2             # contype bits
COL = {
    "base_link": [("box", "0.273 0.233 0.08", "pos", "0 0 0.071"),
                  ("box", "0.059 0.059 0.1485", "pos", "-0.014 0 0.2315")],
    "body5": [("box", "0.072 0.072 0.154", "pos", "0 0 -0.154"),
              ("box", "0.058 0.26 0.057", "pos", "0.008 0 0.057"),
              ("capsule", "0.013", "fromto", "0.041 0 0.12 0.041 0 0.59"),
              ("box", "0.0505 0.025 0.0415", "pos", "0.0735 0 0.6315")],
    "link2": [("capsule", "0.028", "fromto", "0 0 0 -0.264 0 0")],
    "link3": [("capsule", "0.028", "fromto", "0.03 0 0.01 0.245 0 0.056")],
    "link4": [("capsule", "0.025", "fromto", "0 0 0 0.068 0 0.0845")],
    "link5": [("capsule", "0.028", "fromto", "0 0 -0.02 0 0 -0.1")],
    "link6": [("box", "0.035 0.08 0.028", "pos", "0.049 0 0")],
    "link7": [("box", "0.026 0.006 0.028", "pos", "0.045 0.006 0")],
    "link8": [("box", "0.026 0.006 0.028", "pos", "0.045 -0.006 0")],
}


def f(s):
    return [float(v) for v in s.split()]


def main(vec=False):
    root = ET.parse(URDF).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)

    out, act, sens = [], [], []

    def col_for(name):
        if name in COL:
            return COL[name], BODY, 0
        if name.startswith("left_link1"):
            return COL.get("link" + name[-1], []), LEFT, LEFT | RIGHT | BODY
        if name.startswith("right_link2"):
            return COL.get("link" + name[-1], []), RIGHT, LEFT | RIGHT | BODY
        return [], 0, 0

    def emit(name, joint, ind):
        sp = "  " * ind
        link = links[name]
        if joint is None:
            out.append(f'{sp}<body name="{name}" pos="0 0 {GROUND_Z}">')
            out.append(f'{sp}  <joint name="base_x" type="slide" axis="1 0 0" armature="0.5"/>')
            out.append(f'{sp}  <joint name="base_y" type="slide" axis="0 1 0" armature="0.5"/>')
            out.append(f'{sp}  <joint name="base_yaw" type="hinge" axis="0 0 1" armature="0.06"/>')
            out.append(f'{sp}  <site name="base_site" pos="0 0 0" size="0.01"/>')
        else:
            o = joint.find("origin")
            rpy = o.get("rpy", "0 0 0")
            out.append(f'{sp}<body name="{name}" pos="{o.get("xyz")}" euler="{rpy}">')
            jt, jn = joint.get("type"), joint.get("name")
            if jt in ("revolute", "prismatic"):
                lim = joint.find("limit")
                lo, hi = lim.get("lower"), lim.get("upper")
                axis = joint.find("axis").get("xyz")
                mj = "hinge" if jt == "revolute" else "slide"
                if jn == "joint5":
                    extra = 'armature="2" damping="50"'
                    act.append(f'<position name="{jn}" joint="{jn}" kp="50000" kv="1500" '
                               f'forcerange="-{LIFT_FORCE} {LIFT_FORCE}" ctrlrange="{lo} {hi}"/>')
                elif jt == "prismatic":   # gripper finger
                    extra = 'armature="0.1" damping="5"'
                    act.append(f'<position name="{jn}" joint="{jn}" kp="500" kv="20" '
                               f'forcerange="-100 100" ctrlrange="{lo} {hi}"/>')
                else:
                    k = int(jn[-1]) - 1
                    extra = (f'armature="{ARM_ARMATURE[k]}" damping="{ARM_DAMPING[k]}" '
                             f'frictionloss="{ARM_FRICTION[k]}"')
                    act.append(f'<position name="{jn}" joint="{jn}" kp="{ARM_KP[k]}" kv="{ARM_KD[k]}" '
                               f'forcerange="-{ARM_PEAK_TORQUE[k]} {ARM_PEAK_TORQUE[k]}" ctrlrange="{lo} {hi}"/>')
                out.append(f'{sp}  <joint name="{jn}" type="{mj}" axis="{axis}" range="{lo} {hi}" {extra}/>')
        inr = link.find("inertial")
        I = inr.find("inertia")
        full = " ".join(I.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
        out.append(f'{sp}  <inertial pos="{inr.find("origin").get("xyz")}" '
                   f'mass="{inr.find("mass").get("value")}" fullinertia="{full}"/>')
        geoms, ct, ca = col_for(name)
        for i, (typ, size, key, val) in enumerate(geoms):
            out.append(f'{sp}  <geom name="{name}_c{i}" type="{typ}" size="{size}" {key}="{val}" '
                       f'contype="{ct}" conaffinity="{ca}"/>')
        if name.startswith("wheel"):
            out.append(f'{sp}  <geom type="cylinder" size="{WHEEL_RADIUS} 0.0235" pos="0.03 0 0" '
                       f'euler="0 1.5708 0" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1"/>')
        if vec and geoms and ct in (LEFT, RIGHT):
            # touch volume: reports the normal force of any contact on this arm link
            out.append(f'{sp}  <site name="touch_{name}" type="box" size="0.4 0.4 0.4" group="4" rgba="0 0 0 0"/>')
            sens.append(f'<touch name="touch_{name}" site="touch_{name}"/>')
        if name.endswith("link16") or name.endswith("link26"):
            side = "l" if name.startswith("left") else "r"
            out.append(f'{sp}  <site name="tcp_{side}" pos="{TCP_OFFSET}" size="0.012" rgba="1 0.6 0 1"/>')
        for j in children.get(name, []):
            emit(j.find("child").get("link"), j, ind + 1)
        out.append(f"{sp}</body>")

    emit("base_link", None, 2)
    # links two steps apart in the wrist/gripper overlap by construction; the two fingers meet when closed
    excl = []
    for pre in ("left_link1", "right_link2"):
        for a, b in ((2, 4), (3, 5), (4, 6), (4, 7), (4, 8), (5, 7), (5, 8), (7, 8)):
            excl.append(f'<exclude body1="{pre}{a}" body2="{pre}{b}"/>')
    if vec:
        # Omni wheels as native actuators: each is a torque-limited velocity servo whose transmission is
        # the base twist projected on that wheel (gear = -row / r), so actuator velocity = wheel speed
        # [rad/s] and actuator force = wheel torque [N*m]; traction = torque / r along the rolling direction.
        import math
        for k, j in enumerate(children["base_link"][:4]):
            x, y, _ = f(j.find("origin").get("xyz"))
            yaw = f(j.find("origin").get("rpy"))[2]
            tx, ty = -math.sin(yaw), math.cos(yaw)
            g = [-tx / WHEEL_RADIUS, -ty / WHEEL_RADIUS, 0, 0, 0, -(x * ty - y * tx) / WHEEL_RADIUS]
            lim = j.find("limit")
            act.append(f'<velocity name="wheel{k + 1}" site="base_site" gear="{" ".join(f"{v:.6f}" for v in g)}" kv="2" '
                       f'forcerange="-{lim.get("effort")} {lim.get("effort")}" ctrlrange="-{lim.get("velocity")} {lim.get("velocity")}"/>')
        for a in list(act):
            n = a.split('name="')[1].split('"')[0]
            if "joint1" in n[-7:] or "joint2" in n[-7:] or n.startswith("wheel"):
                if n.startswith("wheel") or n[-1] in "123456":
                    sens.append(f'<actuatorfrc name="frc_{n}" actuator="{n}"/>')
        for sd in "lr":
            sens.append(f'<framepos name="tcp_{sd}_pos" objtype="site" objname="tcp_{sd}"/>')
            sens.append(f'<framequat name="tcp_{sd}_quat" objtype="site" objname="tcp_{sd}"/>')
    xml = f"""<mujoco model="LIFT3">
  <compiler angle="radian" balanceinertia="true"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <default>
    <geom rgba="0.6 0.62 0.66 1" friction="0.8 0.01 0.001"/>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 0.05" contype="{BODY}" conaffinity="0" rgba="0.3 0.32 0.36 1"/>
{chr(10).join(out)}
  </worldbody>
  <contact>
    {(chr(10) + '    ').join(excl)}
  </contact>
  <actuator>
    {(chr(10) + '    ').join(act)}
  </actuator>
  <sensor>
    {(chr(10) + '    ').join(sens)}
  </sensor>
</mujoco>
"""
    path = OUT.replace(".xml", "_vec.xml") if vec else OUT
    with open(path, "w") as fh:
        fh.write(xml)
    print("wrote", path)


if __name__ == "__main__":
    main(False)
    main(True)
