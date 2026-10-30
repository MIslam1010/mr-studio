"""Theory text (HTML) for the closed-chains tab, including the further-reading list."""
from __future__ import annotations

from ...references import further_reading

THEORY = r"""
<h2>Closed chains — cut joint &amp; loop closure</h2>
<span class="ref">Modern Robotics Ch. 7 (7.1 kinematics, 7.2 differential kinematics, 7.3 singularities)
and Sec. 8.7 (constrained dynamics)</span>
<p>The loop is cut open at one pin — the <b>cut joint</b> (yellow ring). What is left is a tree of
serial branches rooted at ground pivots, each described exactly like an open chain: screw axes
$\mathcal{S}_i$, home pose $M$, CoM frames $M_{i-1,i}$, spatial inertias $\mathcal{G}_i$ (gravity $-y$).
Re-joining the two tips gives the loop-closure constraint</p>
$$\phi(\theta)=p_{1}(\theta)-p_{2}(\theta)=0,\qquad A(\theta)=\frac{\partial\phi}{\partial\theta}\in\mathbb{R}^{2\times n}$$
<p>Grübler: $dof=3(N-1-J)+\sum f_i$ = 1 for the four-bar ($N=4,J=4$) and 2 for the five-bar ($N=5,J=5$).</p>
<h3>Forward kinematics: actuated → passive (7.1)</h3>
<p>Closed form: the actuated joints fix the last passive joint of each branch; the cut pin lies on a
circle around each of them → circle–circle intersection with two <b>assembly modes</b> $\sigma=\pm1$.
The app checks it against Newton–Raphson on the same equations (MR 6.2):</p>
$$\theta_p^{k+1}=\theta_p^{k}-A_p^{\dagger}(\theta^{k})\,\phi(\theta^{k})$$
<h3>Differential kinematics (7.2)</h3>
$$A_a\dot{\theta}_a+A_p\dot{\theta}_p=0\quad\Rightarrow\quad\dot{\theta}_p=-A_p^{-1}A_a\dot{\theta}_a\quad(cf.\ 7.15)$$
<p>The columns of $A$ come from each branch's space Jacobian $J_s$ (point velocity
$\dot{p}=v_s+\omega_s\times p$), and $\dot{A}\dot{\theta}$ from $\dot{J}_{si}=[\mathrm{ad}_{\mathcal{V}_{i-1}}]J_{si}$.</p>
<h3>Singularities (7.3)</h3>
<p><b>Actuator singularity</b> (Type II, parallel): $\det A_p=0$ — locked actuators no longer hold the
mechanism. Four-bar: coupler and rocker aligned, transmission angle $\mu=0$ or $180°$ (dead centre
for the crank), since $|\det A_p|=L_2L_3|\sin\mu|$. Five-bar: the two distal links are collinear.<br>
<b>End-effector singularity</b> (Type I, serial): a five-bar branch fully stretched or folded,
$\sin\theta_2=0$ or $\sin\theta_4=0$ — this is the workspace boundary.<br>
<b>Configuration-space singularity</b>: $A$ itself loses rank (change-point four-bar folded flat).</p>
<h3>Grashof condition (four-bar)</h3>
$$s+l\leq p+q$$
<p>$s,l$ = shortest and longest link, $p,q$ the others. If it holds, the shortest link turns fully:
crank-rocker (shortest = crank), double-crank (shortest = ground), double-rocker (shortest = coupler).
Equality: change-point. Otherwise: triple-rocker. Design rule: $40°\leq\mu\leq140°$.</p>
<h3>Constrained dynamics (8.7)</h3>
<p>$M$ is block-diagonal and $h=c+g$ is stacked from the branches (mr.MassMatrix,
mr.VelQuadraticForces, mr.GravityForces). The Lagrange multiplier $\lambda$ is the force of the cut pin
on branch 1 (arrow):</p>
$$M(\theta)\ddot{\theta}+h(\theta,\dot{\theta})=\tau+A^{T}(\theta)\lambda\quad(cf.\ 8.93)$$
<p><b>Sign convention:</b> MR writes $\tau=M\ddot{\theta}+h+A^{T}\lambda$ (8.93, 8.98), so the $\lambda$
used here is $-\lambda$ of MR: here $\lambda$ is directly the force the cut pin exerts on branch 1.</p>
<p>Forward dynamics differentiates the constraint twice, with Baumgarte stabilisation
$\ddot{\phi}+2\alpha\dot{\phi}+\beta^{2}\phi=0$, and solves one $(n+2)\times(n+2)$ KKT system per RK4 stage:</p>
$$M\ddot{\theta}-A^{T}\lambda=\tau-h$$
$$A\ddot{\theta}=-\dot{A}\dot{\theta}-2\alpha A\dot{\theta}-\beta^{2}\phi$$
<p>Inverse dynamics: from $\theta_a(t)$ solve $\theta_p,\dot{\theta}_p,\ddot{\theta}_p$, then with the
selection matrix $S$ of the actuated joints:</p>
$$\left[S\ \ A^{T}\right]\left[\tau_a;\ \lambda\right]=M\ddot{\theta}+h$$
<p>Constraint forces do no work ($\lambda^{T}A\dot{\theta}=0$), hence the power balance shown in the
plots:</p>
$$\tau_a^{T}\dot{\theta}_a=\frac{d}{dt}\left(K+P\right)$$
"""
THEORY += further_reading([
    ('lavalle', '§4.4, from p. 167', 'closed kinematic chains: loop-closure constraints as varieties'),
    ('lavalle', '§7.4, pp. 337–346', 'closed chains; active–passive decomposition (§7.4.2, p. 343)'),
    ('lavalle', '§4.1–4.2, pp. 127–154', 'configuration space and its topology'),
    ('mls', 'Ch. 3 §5.2–5.3, pp. 132–138', 'parallel manipulators; the four-bar linkage (p. 135)'),
    ('featherstone', 'Ch. 8, from p. 141', 'closed-loop dynamics: loop constraints (p. 143), loop forces (p. 148)'),
])
