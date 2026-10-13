"""Bibliography of the reference library used by the theory panels ("Further reading") and the
Help → References… dialog.

``REFERENCES[key]`` = dict(short=..., citation=..., path=..., note=...).  ``path`` is the user's
local PDF (may not exist on another machine -- the dialog then hides the "Open PDF" button).
``cite(key, where)`` returns a short HTML citation such as
``<b>Featherstone</b>, Ch. 5, Table 5.1, p. 96`` for use inside the theory HTML.

Page numbers quoted in the theory panels are the *printed* page numbers of the editions below
(not PDF page indices); each was checked against the rendered page.
"""
from __future__ import annotations

import html
import os

_TB = "/Users/mislam/Documents/Edu/Lectures/Robotics/Textbooks"
_SUPP = f"{_TB}/SBEG461 Supplementary References"

REFERENCES: dict[str, dict] = {
    "mr": dict(
        short="Lynch & Park",
        citation="K. M. Lynch and F. C. Park, <i>Modern Robotics: Mechanics, Planning, and Control</i>, "
                 "Cambridge University Press, 2017.",
        path=f"{_TB}/Modern Robotics/MR-v2.pdf",
        note="The textbook the whole app follows (section numbers in every tab refer to it)."),
    "mls": dict(
        short="Murray–Li–Sastry",
        citation="R. M. Murray, Z. Li and S. S. Sastry, <i>A Mathematical Introduction to Robotic "
                 "Manipulation</i>, CRC Press, 1994.",
        path=f"{_TB}/A Mathematical Introduction to Robotic Manipulation, Richard M. Murray, Zexiang Li, "
             "S. Shankar Sastry, S. Shankara Sa, (1994, CRC Press).pdf",
        note="Screw theory, product of exponentials, Lagrangian dynamics and control, rigorous proofs."),
    "featherstone": dict(
        short="Featherstone",
        citation="R. Featherstone, <i>Rigid Body Dynamics Algorithms</i>, Springer, 2008.",
        path=f"{_SUPP}/Rigid Body Dynamics Algorithms, Featherstone, 2008/"
             "Rigid Body Dynamics Algorithms, Featherstone, 2008.pdf",
        note="Spatial vector algebra; RNEA (Ch. 5), CRBA (Ch. 6), ABA (Ch. 7) — the algorithms inside "
             "Pinocchio and re-implemented in mrstudio/featherstone.py."),
    "astrom_murray": dict(
        short="Åström & Murray",
        citation="K. J. Åström and R. M. Murray, <i>Feedback Systems: An Introduction for Scientists and "
                 "Engineers</i>, Princeton University Press, 2008 (electronic edition v2.10, 2009).",
        path=f"{_SUPP}/Feedback Systems, An Introduction for Scientists and Engineers, 22Feb09.pdf",
        note="Stability, Lyapunov functions, second-order systems, state feedback, PID control."),
    "lavalle": dict(
        short="LaValle",
        citation="S. M. LaValle, <i>Planning Algorithms</i>, Cambridge University Press, 2006.",
        path=f"{_SUPP}/Planning Algorithms, 2006.pdf",
        note="Configuration spaces, closed kinematic chains, path-constrained (time-optimal) trajectories."),
    "gotor": dict(
        short="Pardos-Gotor",
        citation="J. M. Pardos-Gotor, <i>Screw Theory in Robotics: An Illustrated and Practicable "
                 "Introduction to Modern Mechanics</i>, CRC Press, 2021.",
        path=f"{_TB}/Screw Theory in Robotics, An Illustrated and Practicable Introduction to Modern "
             "Mechanics, Jose M Pardos Gotor, (2021, CRC Press).pdf",
        note="Screws, PoE, Paden–Kahan and Pardos-Gotor subproblems for closed-form IK, screw dynamics."),
    "corke": dict(
        short="Corke",
        citation="P. Corke, <i>Robotics, Vision and Control: Fundamental Algorithms in MATLAB®</i>, "
                 "Springer Tracts in Advanced Robotics vol. 73, Springer, Berlin, 2011 (1st edition).",
        path=f"{_TB}/Robotics, vision and control fundamental algorithms in MATLAB&amp;reg;.pdf",
        note="Poses, DH kinematics, numerical IK, Jacobians, trajectories, dynamics with the Robotics "
             "Toolbox (used by the app for DH cross-checks)."),
}


def pdf_path(key: str) -> str | None:
    """Local PDF of reference ``key`` if it exists on this machine, else None."""
    p = REFERENCES[key].get("path")
    return p if p and os.path.exists(p) else None


def cite(key: str, where: str = "") -> str:
    """Short HTML citation, e.g. cite("featherstone", "Ch. 5, Table 5.1, p. 96")."""
    ref = REFERENCES[key]
    s = f"<b>{html.escape(ref['short'])}</b>"
    return f"{s}, {where}" if where else s


def further_reading(items: list[tuple[str, str, str]]) -> str:
    """HTML "Further reading" section for a theory panel.

    items = [(key, where, what), ...]  ->  one line per pointer:  Short name, where — what."""
    rows = "".join(f"<br>• {cite(k, w)} — {t}" for k, w, t in items)
    return (f'<h3>Further reading</h3><p><span class="ref">Printed page numbers; full citations and PDFs: '
            f'Help → References…</span>{rows}</p>')


def references_html() -> str:
    """All references as an HTML list (used by the dialog and the tests)."""
    return "".join(f"<p><b>{html.escape(r['short'])}</b><br>{r['citation']}<br>"
                   f"<span style='color:gray'>{html.escape(r['note'])}</span></p>" for r in REFERENCES.values())
