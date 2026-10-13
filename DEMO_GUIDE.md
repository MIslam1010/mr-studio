# MR Studio: 10-minute demo script

This script is for the presenter, who does not need to be a robotics expert. Each step says
**what to click**, **what to say** (in quotes, adapt freely) and **what the audience should
notice**. The ⭐ marks the "wow" moments. The last part lists likely questions from the professor,
with short, correct answers.

## Before the demo (2 minutes, beforehand)

1. Start the app: double-click `MR_Studio.command` (or run `.venv/bin/python run_app.py`). Maximise the window.
2. Keep the **Theory panel** on (button at the top right, or F1). It shows the textbook equations of each tab.
3. Open the **⑩ Validation** tab once and press **▶ Run all checks**, so everything is loaded and
   warm. Then go back to tab ①.
4. Leave the robot on **UR5 (6R industrial)** and the engine on **Pinocchio**.
5. On a projector, press **F5** (Projector mode): bigger fonts, equations and lines. Press F5 again to go back.
6. Optional: record short clips beforehand for your slides with **● Record** (⌘R), and save PNGs with ⌘⇧S.

---

## The script

### 0:00 – 1:00 · Introduction (tab ① Robot & FK)

> "MR Studio simulates the kinematics and dynamics of robot arms and closed mechanisms. It follows
> *Modern Robotics* by Lynch and Park, chapter by chapter. Every robot is described only by its
> screw axes and its home pose, the product-of-exponentials way. There are no DH tables inside.
> Each tab names the section of the book it implements, and the panel on the right shows the
> equations it uses."

Point at the **space screw axes table** at the bottom, then at the equation
$T(\theta) = e^{[\mathcal{S}_1]\theta_1}\cdots e^{[\mathcal{S}_n]\theta_n}M$ in the theory panel.

### 1:00 – 2:15 · Forward kinematics (tab ①)

* Click **Random** a few times, then **▶ Animate**.
* Tick **Equivalent screw of T·M⁻¹ (Chasles)**. A magenta line appears.
  > "Any rigid displacement is one screw motion, a rotation about this line combined with a
  > translation along it. This is the Chasles–Mozzi theorem from chapter 3."
* Point at the green line under the pose matrix:
  > "The pose is computed twice, in the space form and in the body form of the formula. They
  > agree to about 1e-16."
* ⭐ Switch the robot (top bar) to **PUMA 560 (from Corke DH)**.
  > "This robot exists only as a DH table in Peter Corke's toolbox. The app converted it
  > automatically into screw axes and spatial inertias. The Validation tab proves that the
  > conversion is exact."

  Switch back to **UR5**.

### 2:15 – 3:30 · Velocities, singularities (tab ②)

* Point at the ellipsoid at the tool (shown by default).
  > "This is the manipulability ellipsoid. It shows how easily the tool can move in each direction."
* ⭐ Click **⚠ Find nearest singularity**. The robot moves and the ellipsoid flattens.
  > "At a singularity the Jacobian loses rank. The tool cannot move in one direction, whatever the
  > joints do. That is why the ellipsoid collapses to a flat disc."
* Optionally show the **Statics** box:
  > "The same Jacobian, transposed, converts a force at the tool into joint torques: τ = Jᵀ F."

### 3:30 – 4:45 · Inverse kinematics (tab ③)

* Click **Random reachable**, then **▶ Solve IK**. The transparent ghosts are the iterations.
  > "This is the Newton–Raphson algorithm of section 6.2, the same one as the book's IKinBody.
  > Each ghost is one iteration; it converges in a handful of steps."
* ⭐ Click **Find all solutions**.
  > "A 6-joint arm can usually reach the same pose in several ways (up to 8 for the UR5):
  > elbow up or down, wrist flipped, and so on."
* Optional: switch to **Franka Panda (7R redundant)** and click **▶ Self-motion (null space)**.
  > "This arm has 7 joints for a 6-D task. The elbow can move while the gripper stays perfectly still."

### 4:45 – 6:00 · Dynamics (tab ④)

* Click **Random goal**, then **Compute τ(t) by RNEA**.
  > "These are the joint torques needed to follow this motion, from the recursive Newton–Euler
  > algorithm of chapter 8."
* Click **Simulate**, then **▶ Play**.
  > "Now the opposite problem: we apply torques and let physics decide the motion. That is
  > forward dynamics."
* ⭐ Click **Compare integrators**.
  > "Without friction, energy must be conserved. The simple explicit Euler method creates energy
  > out of nothing, a numerical artefact. The higher-order integrators keep the energy constant."

### 6:00 – 7:15 · Trajectories & control (tab ⑤)

* Click **Generate trajectory** (quintic time scaling), then **Time-optimal (TOPP-RA)**.
  > "TOPP-RA finds the fastest possible timing of the same path that never exceeds the motor torque
  > and speed limits. Notice that at every instant some limit is active. That is what makes it
  > time-optimal."
* ⭐ **Computed torque** is the default controller. Click **Run controller**, then **Compare controllers**.
  > "Computed torque uses the dynamic model to cancel the robot's nonlinear dynamics. The tracking
  > error then behaves like a simple, critically damped spring. PID alone does worse."

### 7:15 – 8:30 · Closed chains and the Stewart platform (tabs ⑥ and ⑦)

* Tab ⑥ (four-bar linkage): click **▶ Rotate crank** and tick **Coupler curves**. Then click
  **Flip assembly mode σ**.
  > "In a closed chain the joints are not independent: the loop must stay closed. We cut the loop
  > open at one joint and enforce the closure as a constraint. For the same crank angle, the
  > linkage can be assembled in two ways. These are the two assembly modes."
* Switch the mechanism box to **Five-bar parallel robot (2 DOF)** and click **▶ P traces a circle**,
  then **▶ Free swing (live RK4)**.
  > "This is now a real-time dynamic simulation of the closed chain. The Lagrange multipliers are
  > exactly the forces in the cut joint (section 8.7). Baumgarte stabilisation stops the loop from
  > drifting open numerically."
* Tab ⑦: click **▶ Dance**.
  > "For this parallel robot the inverse kinematics is easy: the six leg lengths follow from the
  > pose by a formula."
* ⭐ Switch the input box to **Forward kinematics: leg lengths → pose** and tick
  **Show all FK solutions (ghost platforms)**.
  > "The opposite direction is hard. For the same six leg lengths there are several possible
  > platform poses, the ghosts. Newton–Raphson finds them from different starting guesses."

### 8:30 – 10:00 · ⭐ The validation suite (tab ⑩)

* Click **▶ Run all checks**. The table fills in live; it takes about 30 seconds.
  > "How do we know any of this is right? Every quantity is computed by at least two programs
  > that share no code: the book's own Python code, Pinocchio (a C++ library that uses
  > Featherstone's algorithms), Peter Corke's DH-based toolbox, and a Lagrangian derived
  > symbolically with SymPy. We also check laws of physics: energy conservation, the power
  > balance, and the skew-symmetry of Ṁ − 2C."
* Point at the green banner: **"244 / 244 checks passed"** (the exact numbers may differ).
  > "Across all the machine-precision comparisons the worst error is about 1e-12, and most agree
  > to 1e-15. Double precision itself is 2e-16."
* Click the row **Newton–Euler = symbolic Euler–Lagrange — Planar 2R**, and read the details pane:
  > "Newton–Euler works with forces and Lagrange works with energy. Two completely different
  > derivations give the same torques to 15 digits."
* Click **Export report (HTML)…** to save the report you can leave with the professor.
* Closing line:
  > "Every formula follows the book, every number is checked against independent code and
  > physics, and you can rerun the whole proof with one click."

---

## Bonus segments (if there is time or the professor asks)

* **⑧ Task-space & Force** — choose **Hybrid motion–force**, click **Run simulation**, then **▶ Play**.
  > "The arm wipes a surface: it controls the motion along the surface and the force into it at the same
  > time, using the projection matrix P(θ) of section 11.6."
  Then try **Impedance** with and without the wrist force sensor: "with the sensor we can make the arm
  feel like any mass-spring-damper we choose."
* **⑨ 3×RPR & Delta** — click **6 real FK solutions**.
  > "This is the book's first parallel-robot example (7.1.1). For the same leg lengths there are six
  > different poses — all six are drawn." Switch to **Delta** and click **▶ Pick & place**.
* **⑦ Stewart–Gough → Dynamics** — compare dynamic and static leg forces; **▶ Simulate push** shows
  that without feedback the platform is unstable, with the leg servo it returns home.
* **File → Import URDF…** (⌘O) — pick an example (e.g. a KUKA). "Any robot described in URDF — the
  standard ROS format — is converted automatically to screw axes and validated against Pinocchio."
  If the professor has a URDF of a lab robot, load it live.
* Robots with **prismatic joints**: choose **Stanford arm (RRPRRR)** or **SCARA Cobra 600** in the top bar.

### The SBEG 361/461 course robots (top-bar robot box)

Six robots from the course materials are built in. They use exactly the numbers in the shared course file
(`_build/shared/course_robots.py`, copied into `mrstudio/course_models.py`; a test checks that the two match):

| Robot box entry | What it is | Good demo |
|---|---|---|
| **Course arm (myCobot-280 class 6R)** | the lab arm, MR Example 4.5 layout, reach ≈ 0.28 m | ① FK: same screw table as the lab sheet; ④ Dynamics: torques of a few tenths of N·m, all inside the servo ratings |
| **EOD arm (6R on UGV)** | 461 project S5 arm with dynamics | ④: gravity dominates joint 2 (shoulder) |
| **EOD arm (3R planar)** | 461 Weeks 1–3 vertical-plane arm | ④ at θ = 0: g(0) = (106.44, 32.86, 3.43) N·m, as in the notes |
| **Arm–exoskeleton 2R** | shoulder + elbow in the sagittal plane | ③ IK / ④ gravity compensation |
| **Rehab robot 2R (planar)** | horizontal-plane rehab robot (gravity ⟂ plane) | ④: g(θ) = 0, only inertia and Coriolis remain |
| **Hapkit (1-DOF haptic paddle)** | 1-DOF paddle, handle 90 mm | ④: a pendulum, M = I_zz + m d² |

> "These are the robots the students use in the labs. The app gives the same numbers as their lab
> sheets, and the Validation tab checks them against Pinocchio like every other robot."

All inertial data and rated torques for these robots are *illustrative* teaching values, as in the course file.

## Likely questions, with short answers

**Why product of exponentials instead of Denavit–Hartenberg?**
PoE needs only two frames, {s} and {b}, plus one screw axis per joint. There are no frame-assignment
rules or special cases for parallel axes, and revolute and prismatic joints are treated the same
way. DH needs a frame on every link, chosen by convention. The app still imports DH models and
proves the conversion is exact.

**How do you know the results are correct?**
Independent implementations agree to about 1e-12 or better (most to 1e-15), and physical laws hold (energy, power,
passivity). The test-suite also contains *negative controls*: if we change one link's inertia by
1 %, the checks fail. So the tests can actually catch errors.

**Why does explicit Euler drift?**
It is only first-order accurate and does not preserve energy. For an oscillator, each step follows
the tangent of the energy curve, so the state spirals outward and energy is gained. The error grows
with the time step. Adaptive RK45 with tight tolerances keeps the relative energy drift around 1e-10.

**What does "Ṁ − 2C is skew-symmetric" mean?**
If $C$ is built from the Christoffel symbols of the mass matrix, then $x^T(\dot M - 2C)x = 0$ for
every vector $x$. Physically it means the Coriolis and centripetal forces do no work: they only
redistribute kinetic energy. This "passivity" property is the basis of stability proofs for robot
controllers (MR Prop. 8.1, Ch. 11).

**What is a singularity?**
A configuration where the Jacobian loses rank. The tool cannot move in some direction, whatever the
joint speeds. Inverse kinematics near it needs very large joint speeds. Statically, the arm can
resist forces in that direction without any motor torque. The manipulability ellipsoid flattens.

**Space Jacobian vs body Jacobian?**
They describe the same velocity in two frames: the fixed frame {s} and the tool frame {b}. They
are related by the adjoint map, $J_b = [\mathrm{Ad}_{T_{bs}}]J_s$. The suite checks this.

**How are closed chains handled?**
The loop is cut at one joint, which leaves a tree of serial chains. Loop closure becomes the
constraint $\phi(\theta) = 0$, with Jacobian $A(\theta)$. The dynamics are
$M\ddot\theta + h = \tau + A^T\lambda$. The Lagrange multipliers $\lambda$ are the constraint forces
in the cut joint. The resulting linear system is solved, with Baumgarte stabilisation so the loop
does not drift open (MR Ch. 7, Sec. 8.7).

**Why does the Stewart platform's forward kinematics have multiple solutions?**
Given six leg lengths, finding the platform pose means solving a system of polynomial equations.
The general Stewart–Gough platform has up to 40 (complex) solutions. Newton–Raphson finds the one
nearest its starting guess, so different starting guesses reveal different real assembly modes.
Inverse kinematics, by contrast, is a closed-form formula.

**Why is Pinocchio so much faster than the textbook code?**
It is written in C++ and uses $O(n)$ recursive algorithms, such as the articulated-body algorithm
for forward dynamics, which never forms the mass matrix. Both engines are built from the same
screw-axis data and agree to machine precision. The engine selector in the top bar switches between them.

**Pinocchio uses different conventions. How can they agree?**
Pinocchio stores twists as $(v, \omega)$ and the book uses $(\omega, v)$. The suite converts
explicitly: Pinocchio's LOCAL-frame Jacobian equals the book's body Jacobian with its two 3-row
blocks swapped.

**Why are the IK tolerances 1e-7 and not 1e-15?**
The book's `MatrixLog3` computes the rotation angle with arccos, which cannot resolve rotations
smaller than about 1.5e-8 rad in double precision. That is the attainable accuracy of any IK built
on it, and the suite states this openly.

**What is computed torque?**
Feedback linearisation: $\tau = M(\theta)(\ddot\theta_d + K_p e + K_i\int e + K_d\dot e) + h(\theta,\dot\theta)$ (MR Eq. 11.35).
With a perfect model the error obeys $\ddot e + K_d\dot e + K_p e = 0$ (with $K_i = 0$), a linear
spring-damper whose gains you choose.

**Why does pure force control let the arm slide sideways?**
Force control only regulates the force *into* the surface; the directions along the surface are not
controlled at all, so any model error makes the end-effector drift. That is exactly why hybrid
motion–force control (MR 11.6) controls motion in those directions and force in the normal one.

**How does URDF import work?**
Pinocchio parses the file; the app reads the screw axes from the space Jacobian at θ = 0, M from the
tip link, and the CoM frames and spatial inertias from the inertial elements (MR 8.8). The Validation
tab then checks FK, inverse dynamics and the mass matrix against Pinocchio's own model of the same file.

**What is TOPP-RA?**
Time-Optimal Path Parameterisation by Reachability Analysis (Pham & Pham, 2018). For a fixed
geometric path, it finds the fastest timing that respects the torque and velocity limits. The
result is "bang-bang": at every instant at least one limit is active.

**Can it handle my own robot?**
Yes. In the **Robot Builder** (top bar) you enter each joint's axis $\omega_i$, a point $q_i$ on it,
and the home pose $M$. The app generates plausible inertias, and the new robot appears in every
tab. It is also included in the next validation run.
