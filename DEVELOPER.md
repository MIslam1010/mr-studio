# MR Studio — developer guide

Desktop app (PySide6 + pyqtgraph OpenGL) that simulates kinematics and dynamics of open and closed
robotic chains **strictly following _Modern Robotics_ (Lynch & Park, 2017)**: screw axes, product of
exponentials, twists/wrenches, adjoints, Newton–Euler, etc. Audience: a robotics professor. The
user is not an expert; the app must look polished and every number must be verifiable.

Run: `.venv/bin/python run_app.py` (from the project root). Python 3.11 venv in `.venv`.

Screenshot (renders the window, saves PNGs, exits — use it to visually check your work):
```
.venv/bin/python run_app.py --screenshot <dir> --tab <index> --only [--robot "Planar 3R arm"] [--delay 2500]
```
Tab indices (`--tab`, 0-13): 0 FK, 1 Velocity, 2 IK, 3 Dynamics, 4 Control, 5 Closed chains, 6 Stewart, 7 Task-space,
8 Parallel, 9 Planning, 10 Haptics, 11 Hardware, 12 Practice, 13 Validation. Keyboard: Ctrl+1...9 jump to tabs 1-9,
Ctrl+Alt+Left/Right go to the previous/next tab. The tab titles carry circled numbers (1-14), so tab N has title number N+1.
Tests: `pip install -r requirements.txt -r requirements-dev.txt`, then `QT_QPA_PLATFORM=offscreen python -m pytest -q`.
`MRSTUDIO_NO_GUI_TESTS=1` skips the GUI tests; `MRSTUDIO_PYTEST_HARD_EXIT=1` ends the run with `os._exit` (old teardown workaround, off by default). On Linux, PySide6 6.12.0's wheel aborts Python 3.11 (`none_dealloc`); use PySide6 6.11.3 there, as CI does (see `tests/conftest.py`).
Tests that compare with the course materials look for them at `MRSTUDIO_COURSE_SHARED` (course_robots.py) and `MRSTUDIO_COURSE_SRC`.
Course robots (SBEG 361/461) live in `mrstudio/course_models.py`; hardware drivers in `mrstudio/hardware/`; installers in `packaging/`.
Look at the PNG with the Read tool. A broken tab shows its traceback inside the tab (see `gui/app.py`).

## Principles
* **Don't reinvent the wheel**: use `modern_robotics` (import as `mr`) for every textbook formula
  (FKinSpace/Body, JacobianSpace/Body, IKinSpace/Body, MatrixExp6/Log6, Adjoint, InverseDynamics,
  MassMatrix, VelQuadraticForces, GravityForces, EndEffectorForces, ForwardDynamics,
  Cubic/QuinticTimeScaling, JointTrajectory, ScrewTrajectory, CartesianTrajectory, ComputedTorque, …),
  Pinocchio (`pinocchio`) for fast dynamics, `roboticstoolbox` (Corke) for DH cross-checks,
  `toppra` for time-optimal path parameterisation, SciPy for ODEs/root finding/LP.
* MR conventions: twists V = (ω, v), wrenches F = (m, f), screw axes in {s} unless stated, joint
  angles in **radians** internally, degrees on screen.
* Each tab cites the MR section it implements and has a theory panel (equations).
* Each core module you write exposes `validation_checks() -> list[mrstudio.checks.Check]` with
  quantitative self-consistency/cross-library checks (these appear in the Validation tab).
* Keep the UI responsive: any computation > ~0.3 s runs in a worker (`QtCore.QThread` or a
  `QTimer`-chunked loop) or is precomputed once and then animated with a `QTimer` (~30 ms).
* Match the style of `mrstudio/gui/tabs/fk_tab.py` (the reference tab). Read it first.

## Package layout
```
mrstudio/
  robots.py       SerialRobot (M, Slist, Blist, Mlist, Glist, joint_limits, tau_max, gravity, planar, home, reach,
                  link_transforms(θ), geometry(θ), com_positions(θ), random_config()), get_robot(name), ROBOTS
                  robots: "Planar 2R arm", "Planar 3R arm", "UR5 (6R industrial)", "PUMA 560 (from Corke DH)",
                  "Franka Panda (7R redundant)"   (all have dynamics).  from_rtb_dh(rtb_robot) converts DH -> PoE.
                  helpers: screw_revolute(w,q,h), trans(p,R), rod_inertia(m,L), screw_axis_line(S)
  kinematics.py   fk_space/fk_body, jacobian_space/body, point_jacobian, task_jacobian, manipulability (Ellipsoid),
                  singular_values, is_singular, joint_torques_from_wrench, ik_newton_raphson (IKResult with history),
                  ik_all_solutions, ik_analytic_planar, null_space_step, workspace_samples, rot_to_rpy, rpy_to_rot
  dynamics.py     DynamicsEngine(robot, backend="pinocchio"|"mr", g=None, Mlist=None, Glist=None):
                    inverse(θ,dθ,ddθ,Ftip=None), mass_matrix(θ), coriolis(θ,dθ) (=c vector), coriolis_matrix(θ,dθ)
                    (pinocchio only), gravity_torque(θ), tip_torque(θ,Ftip), forward(θ,dθ,τ,Ftip=None),
                    kinetic_energy, potential_energy, energy
                  simulate(engine, θ0, dθ0, tau_fn(t,θ,dθ), t_final, dt, method="euler"|"semi-implicit"|"rk4"|"rk45",
                           Ftip=None, with_energy=True, damping=None) -> SimResult(t, theta, dtheta, tau, ke, pe)
                  to_pinocchio(robot) builds a pinocchio.Model from the PoE data.
  checks.py       Check(name, chapter, error, tol, detail, category).passed
  featherstone.py pure-NumPy Featherstone RNEA/CRBA/ABA (Tables 5.1/6.2/7.1) from SerialRobot; validation_checks()
  references.py   REFERENCES bibliography (+ local PDF paths), cite(key, where), further_reading(items) for
                  the theory panels' "Further reading" (printed page numbers, verified); Help -> References…
  gui/
    app.py        MainWindow; TABS list (module, class) — lists all 14 tabs.
    base.py       AppContext (ctx.robot, ctx.backend, ctx.engine(robot=None, backend=None) cached engines,
                  signals robot_changed / backend_changed) and BaseTab.
    viewer.py     SceneView(gl.GLViewWidget) keyed scene graph; RobotGraphic; GhostRobot
    widgets.py    SliderRow, JointSliders, MatrixView, group, hbox, value_label, button, checkbox, combo, spin,
                  plot (pyqtgraph PlotWidget preset), pen(i)
    theory.py     TheoryPanel (HTML with $$display$$ and $inline$ LaTeX via matplotlib mathtext)
    theme.py      colours: ACCENT, GOOD, BAD, WARN, TEXT, MUTED, SERIES (per-joint colours), LINK, JOINT, BASE
    tabs/         one module per tab (fk, velocity, ik, dynamics, control, closed, stewart, advanced_control,
                  parallel, planning, haptics, hardware, practice, validation)
```

## Writing a tab
```python
class MyTab(BaseTab):
    title = "③ Inverse Kinematics"   # escape '&' as '&&'
    uses_serial_robot = True          # False for closed-chain / Stewart tabs (ignores the global robot box)
    def __init__(self, ctx):
        super().__init__(ctx)
        self.view = SceneView()
        controls = QtWidgets.QWidget(); ...            # left column (QVBoxLayout of group(...) boxes)
        bottom = ...                                   # plots / tables under the 3-D view (optional)
        self.assemble(controls, self.view, bottom, THEORY_HTML)
        self.on_robot_changed(ctx.robot)
    def on_robot_changed(self, robot): ...             # rebuild for the new robot (global combo box)
    def on_show(self): ...  /  def on_hide(self): ...  # start / stop timers
```
`ctx.robot` is the robot selected in the top bar. Use `self.view.frame_scene(robot.reach, planar=robot.planar)`.
Planar robots live in the x–y plane with gravity −y; the camera then looks down −z (x right, y up).

### SceneView API (keys are strings; repeated calls update the same item — cheap for animation)
`set_tube(key,a,b,radius,color)`, `set_sphere(key,c,r,color)`, `set_disk(key,c,normal,r,thickness,color)`,
`set_ellipsoid(key,center,axes3x3,radii3,color)`, `set_arrow(key,start,vec,color,radius)`,
`set_polyline(key,pts,color,width,mode="line_strip"|"lines")` (color may be an Nx4 array),
`set_points(key,pts,color,size)` (color may be Nx4), `set_label(key,pos,text,color)`,
`set_triad(key,T,size,label)`, `remove(key)`, `remove_prefix(prefix)`, `set_visible(key,bool)`,
`set_visible_prefix(prefix,bool)`, `frame_scene(extent, planar, center)`.
Colours: "#rrggbb" strings or RGBA float tuples (alpha < 1 => translucent).
`RobotGraphic(view, robot, key="robot", alpha=1.0)` → `.update(θ)` draws the arm; `.show_axes`, `.show_labels`.
`GhostRobot(view, robot, key, color)` → `.update(θ)` translucent stick figure (iterates, alternatives, desired).
Prefix your keys with a tab-specific prefix (e.g. "ik.") so you can `remove_prefix` cleanly.

### Theory panel
HTML with `<h2>`, `<h3>`, `<p>`, `<b>`, `<span class="ref">`. Display math `$$…$$`, inline `$…$`.
matplotlib mathtext subset only: `\theta \dot{x} \ddot{x} \mathcal{V} \mathrm{Ad} \left[ \right] \sum \prod
\frac{a}{b} \sqrt{} \lambda \mu \in \mathbb{R} \cdots \quad \, ^{T} _{i}` — **no** `\begin{...}`, `\text{}`,
`\operatorname`, `\boldsymbol`, `\top`, `\mathbf` ok. Use raw strings (r"""...""").
Verify: `from mrstudio.gui.theory import validate_all; print(validate_all([THEORY]))` must print `[]`.
