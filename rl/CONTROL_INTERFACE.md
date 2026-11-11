# LIFT3 policy control interface

How the trained policy's outputs become motor commands. This describes the simulation used for
training (`rl/lift3_gpu.py`, model built by `rl/build_model.py`).

## Summary

The policy outputs **joint positions** for the two arms and the lift, **wheel speeds** for the
chassis, and **openings** for the grippers. It never outputs torque: a **PD controller** turns
the position targets into motor torque. (Up to run v8 the arms used speed outputs; from run v9
they use position outputs, because speed outputs made the arms shake while holding.)

- Policy rate: 50 Hz (one action every 20 ms)
- Physics / PD rate: 200 Hz (4 physics steps of 5 ms per action)
- Action vector: 19 numbers, each clipped to the range -1 … 1

## Action vector

| Index | Part | Meaning of the output | Scale |
|---|---|---|---|
| 0–3 | Wheels `joint1`–`joint4` | Desired wheel speed | × 20 rad/s |
| 4 | Lift `joint5` | Desired lift height | -1 = 0 m … +1 = 0.38 m |
| 5–10 | Left arm `left_joint11`–`16` | Desired joint position | -1 = lower joint limit … +1 = upper joint limit |
| 11–16 | Right arm `right_joint21`–`26` | Desired joint position | same as left arm |
| 17 | Left gripper | Opening: -1 closed … +1 open | 0 … 44 mm per finger |
| 18 | Right gripper | Opening: -1 closed … +1 open | 0 … 44 mm per finger |

## Arm joints: position command → smoothed target → PD torque

Each control step, for each of the 12 arm joints:

1. **Policy output**: a number in -1 … 1, read as the desired joint position
   (`middle of range + output × half of range`).
2. **Smoothed target**: the PD target does not jump to the commanded position. It travels towards
   it with limited speed and acceleration and brakes in time to stop exactly on it. A steady
   command therefore gives a still arm, and a changed command gives a smooth move without a torque
   spike.

   | Joints | Max target speed | Max target acceleration |
   |---|---|---|
   | 1–3 (shoulder, upper arm, elbow) | 3 rad/s | 6 rad/s² |
   | 4–6 (wrist) | 5 rad/s | 12 rad/s² |

   Target speed towards the command is `min(8 × error, sqrt(2 × max acceleration × error), max speed)`.
3. **Safety clamps**: the target is kept within 0.15 rad of the measured joint position (so it
   cannot run away when the joint is blocked) and inside the joint limits.
4. **PD control** (200 Hz): `torque = kp × (target − position) − kd × velocity`, clamped to the
   motor's peak torque.

   | Joints | kp (N·m/rad) | kd (N·m·s/rad) | Peak torque (N·m) |
   |---|---|---|---|
   | 1–3 | 120 | 6 | 27 |
   | 4–6 | 40 | 2 | 7 |

Joint limits used (rad): joint 1 ±1.57, joint 2 0 … 3.6, joint 3 0 … 3.0, joint 4 ±1.29,
joint 5 ±1.48, joint 6 ±1.74.

## Lift

- Policy output is a desired height between 0 and 0.38 m.
- The height target moves towards it at no more than 33.3 mm/s (URDF limit).
- A PD position servo holds the target: kp 50 000 N/m, kd 1 500 N·s/m, force limit 400 N.

## Wheels (chassis)

- Policy output is a desired speed for each of the four omni wheels, at most 20 rad/s.
- Outputs smaller than 0.05 in size are treated as exactly zero (dead band), so the chassis can
  stand completely still; above that the speed rises linearly to 20 rad/s at an output of 1.
- Each wheel has a velocity loop: `torque = 2.0 × (desired speed − wheel speed)`, clamped to 7 N·m.
- Torque divided by the wheel radius (75 mm) is the traction force along that wheel's rolling
  direction. The four forces move the chassis in x, y and yaw; the chassis has no direct x/y/yaw
  command.

## Grippers

- Policy output is the opening for each gripper; both fingers of a gripper get the same target.
- Finger position servo: kp 500 N/m, kd 20 N·s/m, force limit 100 N.

## Assumptions to check before using this on the real robot

- **PD gains are chosen for the simulation**, not taken from ARX. The real arm's position-mode
  gains should be matched to these (or the simulation changed to match the real ones), otherwise
  real motion and torques will differ from what the policy was trained on.
- **Peak torques** 27 / 27 / 27 / 7 / 7 / 7 N·m match the values in `angle_limit.jpg`.
- **Speed limits** in training (3 and 5 rad/s) are below the motor limits shown in
  `angle_limit.jpg` (5.5 and 20 rad/s).
- **Lift force limit** (400 N) is an assumption: the URDF value (6) cannot hold the platform and
  arms, which weigh about 13.5 kg.
- **Wheel model** assumes no slip, flat ground, and omni wheels with the axle directions given in
  the URDF.
