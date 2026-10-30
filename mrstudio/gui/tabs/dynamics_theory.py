"""Theory-pane HTML for the Dynamics tab (Newton-Euler / Lagrange summary plus further reading)."""
from __future__ import annotations

from ...references import further_reading

THEORY = r"""
<h2>Robot dynamics — Newton–Euler &amp; Lagrange</h2>
<span class="ref">Modern Robotics, Ch. 8 (8.1 Lagrangian, 8.3 Newton–Euler, 8.4 closed form, 8.5 forward dynamics)</span>
<p>Every open chain obeys the same equations of motion (MR Eq. 8.77):</p>
$$\tau=M(\theta)\ddot{\theta}+c(\theta,\dot{\theta})+g(\theta)+J^{T}(\theta)\mathcal{F}_{tip}\quad(8.77)$$
<p>$M$ = mass matrix, $c$ = Coriolis &amp; centripetal torques, $g$ = gravity torques, $\mathcal{F}_{tip}$ = wrench the
end-effector applies to the environment.  The <b>Decomposition</b> plot shows the four terms summing to $\tau$.</p>
<h3>Lagrangian formulation (8.1)</h3>
$$\mathcal{L}=\mathcal{K}-\mathcal{P},\qquad \tau=\frac{d}{dt}\frac{\partial\mathcal{L}}{\partial\dot{\theta}}-\frac{\partial\mathcal{L}}{\partial\theta}\quad(8.15)$$
$$\mathcal{K}=\frac{1}{2}\dot{\theta}^{T}M(\theta)\dot{\theta},\qquad c=\dot{\theta}^{T}\Gamma(\theta)\dot{\theta}\quad(8.14),\ (8.18)$$
$$\Gamma_{ijk}=\frac{1}{2}\left(\frac{\partial m_{ij}}{\partial\theta_k}+\frac{\partial m_{ik}}{\partial\theta_j}-\frac{\partial m_{jk}}{\partial\theta_i}\right)\quad(8.17)$$
<p>Elegant but costly (symbolic derivatives).  The <b>Newton–Euler</b> recursion obtains the same $\tau$ in $O(n)$.</p>
<h3>Recursive Newton–Euler (RNEA, 8.3.2)</h3>
<p>Link frames {i} sit at the centres of mass; $\mathcal{A}_i$ = screw axis of joint $i$ in {i},
$\mathcal{G}_i$ = 6×6 spatial inertia.  Initialise $\mathcal{V}_0=0$, $\dot{\mathcal{V}}_0=(0,\,-g)$,
$\mathcal{F}_{n+1}=\mathcal{F}_{tip}$.</p>
<p><b>Forward iterations</b>  $i=1\ldots n$:</p>
$$T_{i,i-1}=e^{-[\mathcal{A}_i]\theta_i}M_{i,i-1}\quad(8.50)$$
$$\mathcal{V}_i=[\mathrm{Ad}_{T_{i,i-1}}]\mathcal{V}_{i-1}+\mathcal{A}_i\dot{\theta}_i\quad(8.51)$$
$$\dot{\mathcal{V}}_i=[\mathrm{Ad}_{T_{i,i-1}}]\dot{\mathcal{V}}_{i-1}+[\mathrm{ad}_{\mathcal{V}_i}]\mathcal{A}_i\dot{\theta}_i+\mathcal{A}_i\ddot{\theta}_i\quad(8.52)$$
<p><b>Backward iterations</b>  $i=n\ldots 1$:</p>
$$\mathcal{F}_i=[\mathrm{Ad}_{T_{i+1,i}}]^{T}\mathcal{F}_{i+1}+\mathcal{G}_i\dot{\mathcal{V}}_i-[\mathrm{ad}_{\mathcal{V}_i}]^{T}\mathcal{G}_i\mathcal{V}_i\quad(8.53)$$
$$\tau_i=\mathcal{F}_i^{T}\mathcal{A}_i\quad(8.54)$$
<p>The gravity trick: a fictitious upward acceleration $-g$ of the base produces the gravity torques for free.</p>
<h3>Mass matrix from n RNEA calls (8.5)</h3>
$$M(\theta)\,e_j=\mathrm{ID}(\theta,\ \dot{\theta}=0,\ \ddot{\theta}=e_j;\ g=0,\ \mathcal{F}_{tip}=0)$$
<p>That is <b>mr.MassMatrix</b>; Pinocchio uses the Composite-Rigid-Body Algorithm (same matrix, ~1000× faster —
timings in the <b>Mass matrix</b> tab).</p>
<h3>Forward dynamics (8.5)</h3>
$$\ddot{\theta}=M^{-1}(\theta)\left(\tau-c(\theta,\dot{\theta})-g(\theta)-J^{T}(\theta)\mathcal{F}_{tip}\right)\quad(8.81)$$
<p>mr.ForwardDynamics solves this linear system; Pinocchio's Articulated-Body Algorithm is $O(n)$.
The state $(\theta,\dot{\theta})$ is integrated numerically:</p>
<p><b>Explicit Euler</b> (mr.EulerStep):</p>
$$\theta_{k+1}=\theta_k+\Delta t\,\dot{\theta}_k,\qquad \dot{\theta}_{k+1}=\dot{\theta}_k+\Delta t\,\ddot{\theta}_k$$
<p><b>Semi-implicit (symplectic) Euler</b>:</p>
$$\dot{\theta}_{k+1}=\dot{\theta}_k+\Delta t\,\ddot{\theta}_k,\qquad \theta_{k+1}=\theta_k+\Delta t\,\dot{\theta}_{k+1}$$
<p>plus classical RK4 and adaptive RK45 (Dormand–Prince 5(4), SciPy, rtol = 10⁻¹⁰).</p>
<h3>Energy — a built-in correctness test</h3>
$$\mathcal{P}=-\sum_i m_i\,g^{T}p_{c_i}(\theta),\qquad \frac{d}{dt}(\mathcal{K}+\mathcal{P})=\dot{\theta}^{T}(\tau-B\dot{\theta})$$
<p>With $\tau=0$ and no damping the total energy must stay constant: explicit Euler <i>gains</i> energy every step,
the symplectic semi-implicit Euler keeps it bounded, RK4/RK45 conserve it to many digits
(<b>Compare integrators</b>).  With gravity compensation $\tau=g(\theta)$ the robot "floats": $M\ddot{\theta}+c=0$.</p>
<h3>Structural properties (8.1.2)</h3>
$$M(\theta)=M^{T}(\theta)>0,\qquad x^{T}\left(\dot{M}-2C\right)x=0\ \ \forall x$$
<p>Positive definiteness follows from $\mathcal{K}>0$; skew-symmetry of $\dot{M}-2C$ (passivity) is what makes
PD + gravity compensation stable (Ch. 11).  All three are checked live in the <b>Mass matrix</b> tab.</p>
"""
THEORY += further_reading([
    ('featherstone', 'Ch. 2, pp. 7–38', 'spatial vector algebra (spatial inertia §2.13, p. 32)'),
    ('featherstone', '§5.3, pp. 92–97, Table 5.1 p. 96', 'recursive Newton–Euler algorithm'),
    ('featherstone', '§6.2, pp. 104–108, Table 6.2 p. 107', 'composite-rigid-body algorithm (mass matrix)'),
    ('featherstone', '§7.3, pp. 128–131, Table 7.1 p. 132', 'articulated-body algorithm (forward dynamics)'),
    ('mls', 'Ch. 4 §2–3, pp. 156–179', 'Lagrange equations; open chains; Lemma 4.2 (p. 171): M > 0, Ṁ − 2C skew'),
    ('gotor', '§6.2, pp. 152–170; §6.4–6.5, pp. 172–190', 'Lagrange with PoE; recursive Newton–Euler'),
    ('corke', '§9.1, §9.3, pp. 191–204', 'gravity, inertia, Coriolis terms; forward dynamics'),
])
