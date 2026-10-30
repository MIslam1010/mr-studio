"""Theory-pane HTML for the inverse-kinematics tab (MathJax-style markup, rendered by BaseTab.assemble)."""
from __future__ import annotations

from ...references import further_reading

THEORY = r"""
<h2>Inverse kinematics</h2>
<span class="ref">Modern Robotics, Ch. 6: §6.1 analytic IK, §6.2 numerical IK, §6.3 inverse velocity kinematics</span>
<p>Given a desired end-effector configuration $T_{sd}$, find $\theta$ with $T(\theta)=T_{sd}$.  Unlike forward
kinematics there may be <b>no</b> solution, a <b>finite number</b> (up to 8 for UR/PUMA-type arms; a general 6R
arm can have 16) or <b>infinitely many</b> (redundant arms).</p>
<h3>Analytic IK: planar 2R (§6.1)</h3>
<p>Law of cosines for the elbow, then the shoulder:</p>
$$\cos\theta_2=\frac{x^{2}+y^{2}-L_1^{2}-L_2^{2}}{2L_1L_2}$$
$$\theta_1=\mathrm{atan2}(y,x)-\mathrm{atan2}(L_2\sin\theta_2,\ L_1+L_2\cos\theta_2)$$
<p>The two signs of $\theta_2$ are the elbow-down / elbow-up branches.  For the 3R arm with orientation
$\phi$, apply this to the wrist point $(x-L_3\cos\phi,\ y-L_3\sin\phi)$ and set $\theta_3=\phi-\theta_1-\theta_2$.
A PUMA-type 6R arm has up to $2\times2\times2=8$ solutions (shoulder left/right, elbow up/down, wrist flip).</p>
<h3>Newton–Raphson (§6.2.1)</h3>
<p>To solve $x_d-f(\theta)=0$, expand $f$ to first order about the current guess $\theta^{i}$:</p>
$$x_d=f(\theta^{i})+J(\theta^{i})\,\Delta\theta\ \ \Rightarrow\ \ \Delta\theta=J^{\dagger}(\theta^{i})\left(x_d-f(\theta^{i})\right)\quad(6.4),\ (6.6)$$
<p>Near a solution the error is squared at every step: <b>quadratic convergence</b> (the number of correct
digits doubles — the convergence curves bend straight down on the log plot).</p>
<h3>Algorithm IKinBody (§6.2.2)</h3>
<p>For poses the error is the body twist that moves {b} to {d} in unit time:</p>
$$[\mathcal{V}_b]=\log\left(T_{sb}^{-1}(\theta^{i})\,T_{sd}\right),\qquad \theta^{i+1}=\theta^{i}+J_b^{\dagger}(\theta^{i})\,\mathcal{V}_b$$
<p>until $\|\omega_b\|\leq\epsilon_\omega$ and $\|v_b\|\leq\epsilon_v$.  <b>IKinSpace</b> uses
$\mathcal{V}_s=\left[\mathrm{Ad}_{T_{sb}}\right]\mathcal{V}_b$ with $J_s$; since
$J_s=\left[\mathrm{Ad}_{T_{sb}}\right]J_b$ the Newton steps are the same, but it stops on
$\|\omega_s\|\leq\epsilon_\omega$ and $\|v_s\|\leq\epsilon_v$.  $\|\omega_s\|=\|\omega_b\|$, while
$v_s=R\,v_b+p\times R\,\omega_b$ depends on where the {s} origin is, so the two versions may stop at different
iterates.  The app re-runs the official <b>mr.IKinBody / mr.IKinSpace</b> on the same data and prints
$\|\Delta\theta\|$.</p>
<h3>Pseudo-inverse (§6.2.1)</h3>
$$J^{\dagger}=J^{T}\left(JJ^{T}\right)^{-1}\ (n>m),\qquad J^{\dagger}=\left(J^{T}J\right)^{-1}J^{T}\ (n<m)$$
<p>For $n>m$ it returns the minimum-norm $\Delta\theta$ among all that achieve the task.</p>
<h3>Damped least squares</h3>
$$\Delta\theta=J^{T}\left(JJ^{T}+\lambda^{2}I\right)^{-1}\mathcal{V}$$
<p>Each singular direction is scaled by $\sigma_i/(\sigma_i^{2}+\lambda^{2})$ instead of $1/\sigma_i$: steps stay
bounded near singularities, at the price of linear convergence (an extension beyond the MR library).</p>
<h3>Redundancy and self-motion (§6.3)</h3>
<p>For $n>m$ the general solution of $\mathcal{V}=J\dot{\theta}$ is</p>
$$\dot{\theta}=J^{\dagger}\mathcal{V}+\left(I-J^{\dagger}J\right)z$$
<p>The second term lies in the null space of $J$: joints move, the end-effector does not.  The 7-DOF Panda has a
1-D null space for the full pose (elbow swing); the planar 3R arm has one for a position-only task.  A finite step
along $(I-J^{\dagger}J)z$ drifts at second order, so each step is followed by a Newton correction — the drift
readout shows the residual.</p>
<h3>Real-time IK</h3>
<p>"Follow a circle" re-solves IK every 30 ms, warm-started from the previous solution: thanks to quadratic
convergence, one or two iterations per frame suffice.</p>
"""
THEORY += further_reading([
    ('mls', 'Ch. 3 §3.2–3.3, pp. 99–108', 'Paden–Kahan subproblems and closed-form IK'),
    ('gotor', '§4.2–4.4, pp. 53–114', 'numeric vs. geometric IK; Paden–Kahan and Pardos-Gotor subproblems'),
    ('corke', '§7.3, pp. 146–152; §8.4, pp. 187–188', 'closed-form and numerical IK; Jacobian-based iteration'),
])
