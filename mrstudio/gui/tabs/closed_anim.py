"""Animation, free-swing simulation and inverse-dynamics playback for the closed-chains tab (mixin)."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg

from ... import closed_chains as cc
from .. import theme
from ..widgets import pen


class ClosedAnimMixin:
    """Methods mixed into ClosedChainTab (state lives on the tab)."""

    def _stop_anims(self):
        for b in (self.rot_btn, self.circle_btn, self.swing_btn, self.id_play):
            if b.isChecked():
                b.blockSignals(True)
                b.setChecked(False)
                b.blockSignals(False)
        self.rot_btn.setText("▶ Rotate crank")
        self.circle_btn.setText("▶ P traces a circle")
        self.swing_btn.setText("▶ Free swing (live RK4)")
        self.id_play.setText("▶ Play")
        self.timer.stop()
        self.anim = None

    def _anim_toggle(self, kind, on):
        texts = {"crank": ("▶ Rotate crank", "■ Stop"), "circle": ("▶ P traces a circle", "■ Stop"),
                 "swing": ("▶ Free swing (live RK4)", "■ Stop simulation"), "id": ("▶ Play", "■ Stop")}
        btns = {"crank": self.rot_btn, "circle": self.circle_btn, "swing": self.swing_btn, "id": self.id_play}
        if not on:
            btns[kind].setText(texts[kind][0])
            self.timer.stop()
            self.anim = None
            if kind == "swing" and self.theta is not None:
                self.dyn_lbl.setText(self.dyn_lbl.text() + "\n(stopped — the mechanism stays in the last simulated pose)")
            return
        self._stop_anims()
        btns[kind].blockSignals(True)
        btns[kind].setChecked(True)
        btns[kind].blockSignals(False)
        btns[kind].setText(texts[kind][1])
        if self.theta is None:
            self._stop_anims()
            return
        self.anim = kind
        self.trace = []
        self.anim_t = 0.0
        if kind == "circle":
            self.circ_c = self.mech.tip(self.theta)[:2] - np.array([0.05, 0.0])
        elif kind == "swing":
            if not self._start_swing():
                self._stop_anims()
                return
        elif kind == "id":
            if self.id_res is None:                 # compute first (background), play when it is ready
                self._stop_anims()
                self._compute_id(play=True)
                return
            self.bottom.setCurrentIndex(2)
            self.id_k = 0
        self._last = time.time()
        self.timer.start(30)

    def _tick(self):
        try:
            now = time.time()
            dt = min(now - self._last, 0.1)
            self._last = now
            if self.anim == "crank":
                self._tick_crank(dt)
            elif self.anim == "circle":
                self._tick_circle(dt)
            elif self.anim == "swing":
                self._tick_swing()
            elif self.anim == "id":
                self._tick_id()
        except Exception as e:  # never let a timer callback crash the app
            self._stop_anims()
            self.dyn_lbl.setText(f"stopped: {type(e).__name__}: {e}")

    def _tick_crank(self, dt):
        m = self.mech
        w = 2 * np.pi * 0.25 * self.anim_dir
        th1 = self.crank.value() + w * dt
        if not m.crank_feasible(th1):
            # input limit (det A_p = 0): reverse the crank and continue on the other assembly mode
            self.anim_dir *= -1
            th1 = self.crank.value()
            self.sigma_box.blockSignals(True)
            self.sigma_box.setCurrentIndex(1 if self.sigma == 1 else 0)
            self.sigma_box.blockSignals(False)
            self.sigma = -self.sigma
            self._update_kin_plots_four()
        self.crank.set_value(float(cc.wrap(th1)))
        self._kin_update()
        self.trace.append(m.coupler_point(self.theta))
        self.trace = self.trace[-600:]

    def _tick_circle(self, dt):
        self.anim_t += dt
        a = 2 * np.pi * self.anim_t / 4.0
        P = self.circ_c + 0.05 * np.array([np.cos(a), np.sin(a)])
        self.px.set_value(P[0])
        self.py.set_value(P[1])
        self._p_changed()
        if self.theta is not None:
            self.trace.append(self.mech.tip(self.theta))
            self.trace = self.trace[-400:]

    # -- free swing -----------------------------------------------------------------------
    def _tau_vec(self):
        tau = np.zeros(self.mech.n)
        tau[self.mech.actuated[0]] = self.tau1.value()
        if self.mech.na > 1:
            tau[self.mech.actuated[1]] = self.tau3.value()
        return tau

    def _start_swing(self) -> bool:
        m = self.mech
        m.backend = self.ctx.backend
        th0 = self.theta.copy()
        self.sim = dict(t=0.0, th=th0, dth=np.zeros(m.n), W=0.0, D=0.0)
        self.buf = {k: [] for k in ("t", "ke", "pe", "E", "Ec", "phi", "lx", "ly")}
        self.lam_max = 1e-9
        self.bottom.setCurrentIndex(1)
        self.mode_tabs.setCurrentIndex(1)
        return True

    def _tick_swing(self):
        m, s = self.mech, self.sim
        dt = self.dt.value()
        steps = max(1, int(round(0.03 / dt)))
        a, b = (self.alpha.value(), self.beta.value()) if self.cb_baum.isChecked() else (0.0, 0.0)
        tau = self._tau_vec()
        t0 = time.time()
        for k in range(steps):
            th, dth, lam, dW, dD = m.rk4_step(s["th"], s["dth"], s["t"], dt, lambda t, q, dq: tau,
                                                a, b, self.damp.value())
            s.update(th=th, dth=dth, t=s["t"] + dt, W=s["W"] + dW, D=s["D"] + dD)
            if time.time() - t0 > 0.06:   # slow (MR) back-end: stay responsive
                break
        ke, pe = m.energy(s["th"], s["dth"])
        phi = np.linalg.norm(m.constraint(s["th"])[0])
        B = self.buf
        for key, val in (("t", s["t"]), ("ke", ke), ("pe", pe), ("E", ke + pe), ("Ec", ke + pe - s["W"] + s["D"]),
                         ("phi", max(phi, 1e-18)), ("lx", lam[0]), ("ly", lam[1])):
            B[key].append(val)
            del B[key][:-1500]
        t = np.array(B["t"])
        for c, key in zip(self.E_curves, ("ke", "pe", "E", "Ec")):
            c.setData(t, np.array(B[key]))
        self.phi_curve.setData(t, np.array(B["phi"]))
        lx, ly = np.array(B["lx"]), np.array(B["ly"])
        for c, y in zip(self.lam_curves, (lx, ly, np.hypot(lx, ly))):
            c.setData(t, y)
        self.lam_max = max(self.lam_max, float(np.hypot(*lam)))
        self.lam_show = (lam, 0.12 / self.lam_max)
        self.theta = s["th"]
        self._redraw()
        drift = B["Ec"][-1] - B["Ec"][0]
        self.dyn_lbl.setText(f"t = {s['t']:.2f} s   ({m.backend} blocks, dt = {dt * 1e3:.1f} ms)\n"
                             f"K + P − W + D drift = {drift:+.2e} J\n‖φ‖ = {phi:.1e} m    |λ| = {np.hypot(*lam):.2f} N")

    # -- inverse dynamics -----------------------------------------------------------------------
    def _compute_id(self, play: bool = False):
        """Inverse dynamics along the current crank / circle trajectory, in a background thread on a private
        copy of the mechanism (its dynamics engines are not shared with the GUI thread)."""
        m = self.mech
        if m is None or self.theta is None:
            return
        m.backend = self.ctx.backend
        wm = self._make_mech()                       # same parameters, own engine cache
        wm.backend = m.backend
        four = self.four
        if four:
            args = (self.omega.value(), self.sigma, float(self.theta[0]))
            full = m.crank_fully_rotates()
        else:
            c = m.tip(self.theta)[:2] - np.array([self.radius.value(), 0.0])
            args = (c, self.radius.value(), self.period.value(), self.elbows)
        t0 = time.time()

        def job(progress, cancelled):
            if four:
                t, TH, DTH, DDTH = wm.crank_trajectory(*args, N=181)
                traj = lambda tt: wm.crank_trajectory(*args, t=tt)[1:]
                x = np.degrees(np.unwrap(TH[:, 0])) if full else t
                xl = "crank angle θ₁ [°]" if full else "t [s]"
            else:
                t, TH, DTH, DDTH = wm.circle_trajectory(*args, N=181)
                traj = lambda tt: wm.circle_trajectory(*args, t=tt)[1:]
                x, xl = t, "t [s]"
            return wm.inverse_dynamics_trajectory(t, TH, DTH, DDTH, x, xl, traj_fn=traj, cancelled=cancelled,
                                                  progress=progress)

        self.id_btn.setEnabled(False)
        self.dyn_lbl.setText("inverse dynamics (background thread) …")
        self.id_jobs.run(job, on_done=lambda res, req=m: self._id_done(req, res, time.time() - t0, play),
                         on_error=self._id_failed,
                         on_progress=lambda f: self.dyn_lbl.setText(
                             f"inverse dynamics (background thread) … {100 * f:.0f} %"),
                         label="inverse dynamics")

    def _id_failed(self, err):
        self.id_btn.setEnabled(True)
        self.dyn_lbl.setText(f"⚠ {err.strip().splitlines()[-1]}")

    def _cancel_id(self):
        if self.id_jobs.is_busy():
            self.id_jobs.cancel()
        if hasattr(self, "id_btn"):
            self.id_btn.setEnabled(True)

    def _id_done(self, req, res, seconds, play):
        self.id_btn.setEnabled(True)
        m = self.mech
        if req is not m or self.theta is None:      # mechanism rebuilt meanwhile: stale result
            return
        t, x, xl = res.t, res.x, res.xlabel
        self.id_res = res
        for p in (self.ptau, self.pidlam, self.ppow):
            p.clear()
            p.setLabel("bottom", xl)
        names = ["τ₁ (crank)"] if self.four else ["τ₁ (left motor)", "τ₃ (right motor)"]
        for i in range(m.na):
            self.ptau.plot(x, res.tau[:, i], pen=pen(i, 2.4), name=names[i])
        lam = res.lam
        self.pidlam.plot(x, lam[:, 0], pen=pen(0, 1.8), name="λx")
        self.pidlam.plot(x, lam[:, 1], pen=pen(1, 1.8), name="λy")
        self.pidlam.plot(x, np.hypot(lam[:, 0], lam[:, 1]), pen=pen(5, 2.4), name="|λ|")
        self.ppow.plot(x, res.power, pen=pen(2, 3), name="τₐᵀθ̇ₐ (motor power)")
        self.ppow.plot(x, res.dEdt, pen=pen(3, 1.6, "dash"), name="dE/dt (central differences)")
        self.id_cursors = [p.addLine(x=x[0], pen=pg.mkPen(theme.WARN, width=1))
                           for p in (self.ptau, self.pidlam, self.ppow)]
        self.lam_scale_id = 0.10 / max(float(np.hypot(lam[:, 0], lam[:, 1]).max()), 1e-9)
        err = np.abs(res.power - res.dEdt).max() / max(np.abs(res.power).max(), 1e-9)
        self.dyn_lbl.setText(f"Inverse dynamics ({len(t)} samples, {m.backend} blocks, {seconds:.2f} s)\n"
                             f"peak |τ| = {np.abs(res.tau).max():.2f} N·m,  peak |λ| = "
                             f"{np.hypot(lam[:, 0], lam[:, 1]).max():.1f} N\n"
                             f"power balance: max|τθ̇ − dE/dt| / max|τθ̇| = {err:.1e}")
        self.bottom.setCurrentIndex(2)
        if play:
            self._anim_toggle("id", True)

    def _tick_id(self):
        r = self.id_res
        k = self.id_k % len(r.t)
        self.id_k += 1
        self.theta = r.theta[k].copy()
        self.lam_show = (r.lam[k], self.lam_scale_id)
        for c in self.id_cursors:
            c.setValue(r.x[k])
        if self.four:
            self.trace.append(self.mech.coupler_point(self.theta))
        else:
            self.trace.append(self.mech.tip(self.theta))
        self.trace = self.trace[-len(r.t):]
        self._redraw()
