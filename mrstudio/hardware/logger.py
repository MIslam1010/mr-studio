"""Joint logs in the exact CSV formats of the two project starters.

* **SBEG 361** ``starter/hwcompare.py`` (``write_log`` / ``read_log``)::

      t, cmd_1, ..., cmd_n, meas_1, ..., meas_n[, stop]

  t in s, angles in **degrees**, every number written ``%.4f``; ``stop`` = 0 before and 1 from the
  instant the stop command is given (integer, optional; needed for HW-6).

* **SBEG 461** ``starter/sbeg461_project/hardware.py`` (``write_log`` / ``load_log``)::

      t, cmd_1, ..., cmd_n, meas_1, ..., meas_n [, cur_1, ..., cur_n] [, force]

  **SI units** (s, rad), full float precision (``csv.writer`` of ``ndarray.tolist()``); no stop column
  (the 461 ``stop_time(t, qdot, t_trigger, ...)`` takes the trigger time as an argument -- MR Studio
  shows it in the report and in the file name hint).

``HardwareLog`` collects samples in radians during a run and writes either format.  The course formats
stay byte-identical to the starters; MR Studio's diagnostics go to two sidecar files next to the log
(``save_sidecars``):

* ``<log>.meta.json`` -- port, baud, calibration (sign, offset), SDK version, firmware speed %, the event
  list (connect, start, stall, stop, errors, ...) and a summary (stale reads, stalls);
* ``<log>.diag.csv``  -- per row: t, tick dt, fresh (1 = new reading, 0 = repeated after a failed read),
  stop, raw servo degrees sent (``servo_cmd_j``) and read (``servo_meas_j``, empty when stale).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

FORMATS = {"361": "SBEG 361 hwcompare.py (deg, stop column)",
           "461": "SBEG 461 hardware.py (SI: rad)"}


# ------------------------------------------------------------------------------------------- 361
def write_log_361(path, t, cmd, meas, stop=None):
    """Byte-for-byte the 361 ``hwcompare.write_log``: angles in degrees."""
    t = np.asarray(t, float)
    cmd, meas = np.atleast_2d(cmd), np.atleast_2d(meas)
    n = cmd.shape[1]
    header = ["t"] + [f"cmd_{i+1}" for i in range(n)] + [f"meas_{i+1}" for i in range(n)]
    if stop is not None:
        header.append("stop")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for k in range(len(t)):
            row = [f"{t[k]:.4f}"] + [f"{x:.4f}" for x in cmd[k]] + [f"{x:.4f}" for x in meas[k]]
            if stop is not None:
                row.append(int(stop[k]))
            w.writerow(row)


def read_log_361(path) -> dict:
    """The 361 ``hwcompare.read_log``: dict t [N], cmd, meas [N x n] (deg), stop [N] or None."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    data = np.array([[float(x) for x in r] for r in rows[1:] if r], float)
    col = {h: i for i, h in enumerate(header)}
    cmd_cols = sorted((h for h in header if h.startswith("cmd_")), key=lambda h: int(h[4:]))
    meas_cols = sorted((h for h in header if h.startswith("meas_")), key=lambda h: int(h[5:]))
    if len(cmd_cols) != len(meas_cols) or not cmd_cols:
        raise ValueError("log needs matching cmd_i and meas_i columns")
    return dict(t=data[:, col["t"]],
                cmd=data[:, [col[h] for h in cmd_cols]],
                meas=data[:, [col[h] for h in meas_cols]],
                stop=data[:, col["stop"]] if "stop" in col else None)


# ------------------------------------------------------------------------------------------- 461
def write_log_461(path, t, cmd, meas, cur=None, force=None) -> None:
    """Identical to the 461 ``hardware.write_log`` (SI units)."""
    t = np.asarray(t, float)
    cmd = np.asarray(cmd, float).reshape(len(t), -1)
    meas = np.asarray(meas, float).reshape(len(t), -1)
    n = cmd.shape[1]
    cols = [t[:, None], cmd, meas]
    header = ["t"] + [f"cmd_{j + 1}" for j in range(n)] + [f"meas_{j + 1}" for j in range(n)]
    if cur is not None:
        cols.append(np.asarray(cur, float).reshape(len(t), -1))
        header += [f"cur_{j + 1}" for j in range(n)]
    if force is not None:
        cols.append(np.asarray(force, float).reshape(len(t), 1))
        header += ["force"]
    data = np.hstack(cols)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(data.tolist())


def read_log_461(path) -> dict:
    """The 461 ``hardware.load_log``: t (N,), cmd, meas (N, n) in SI, cur / force when present; rows
    sorted by time, duplicate time stamps dropped."""
    path = Path(path)
    with open(path, newline="") as fh:
        header = [h.strip() for h in next(csv.reader(fh))]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    col = {h: i for i, h in enumerate(header)}
    if "t" not in col:
        raise ValueError(f"{path}: column 't' missing")
    order = np.argsort(data[:, col["t"]], kind="stable")
    data = data[order]
    _, keep = np.unique(data[:, col["t"]], return_index=True)
    data = data[np.sort(keep)]

    def group(prefix):
        names = sorted((h for h in header if h.startswith(prefix + "_")), key=lambda h: int(h.split("_")[1]))
        return data[:, [col[h] for h in names]] if names else None

    out = dict(t=data[:, col["t"]], cmd=group("cmd"), meas=group("meas"))
    if out["cmd"] is None or out["meas"] is None or out["cmd"].shape != out["meas"].shape:
        raise ValueError(f"{path}: need matching cmd_j and meas_j columns")
    cur = group("cur")
    if cur is not None:
        out["cur"] = cur
    if "force" in col:
        out["force"] = data[:, col["force"]]
    return out


def read_log(path, fmt: str = "361") -> dict:
    """Either format, converted to radians: dict t, cmd, meas (rad), stop (or None)."""
    if fmt == "361":
        d = read_log_361(path)
        return dict(t=d["t"], cmd=np.radians(d["cmd"]), meas=np.radians(d["meas"]), stop=d["stop"])
    if fmt == "461":
        d = read_log_461(path)
        return dict(t=d["t"], cmd=d["cmd"], meas=d["meas"], stop=None, cur=d.get("cur"))
    raise ValueError(f"unknown log format {fmt!r}")


def load_trajectory(path, fmt: str = "361"):
    """Commanded trajectory of a log (or of a hand-written CSV with t and cmd_j columns): (t, q [rad]).
    The time axis is shifted to start at 0."""
    if fmt == "361":
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        header = [h.strip() for h in rows[0]]
        data = np.array([[float(x) for x in r] for r in rows[1:] if r], float)
        col = {h: i for i, h in enumerate(header)}
        names = sorted((h for h in header if h.startswith("cmd_")), key=lambda h: int(h[4:]))
        if "t" not in col or not names:
            raise ValueError(f"{path}: need columns t and cmd_1 … cmd_n")
        t, q = data[:, col["t"]], np.radians(data[:, [col[h] for h in names]])
    elif fmt == "461":
        with open(path, newline="") as fh:
            header = [h.strip() for h in next(csv.reader(fh))]
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        col = {h: i for i, h in enumerate(header)}
        names = sorted((h for h in header if h.startswith("cmd_")), key=lambda h: int(h.split("_")[1]))
        if "t" not in col or not names:
            raise ValueError(f"{path}: need columns t and cmd_1 … cmd_n")
        t, q = data[:, col["t"]], data[:, [col[h] for h in names]]
    else:
        raise ValueError(f"unknown log format {fmt!r}")
    order = np.argsort(t, kind="stable")
    t, q = t[order], q[order]
    _, keep = np.unique(t, return_index=True)
    keep = np.sort(keep)
    return t[keep] - t[keep][0], q[keep]


# ------------------------------------------------------------------------------------------- log
class HardwareLog:
    """Samples of one run (radians internally)."""

    def __init__(self, n: int):
        self.n = n
        self._t: list[float] = []
        self._cmd: list[np.ndarray] = []
        self._meas: list[np.ndarray] = []
        self._stop: list[int] = []
        self._fresh: list[int] = []
        self._dt: list[float] = []
        self._servo_cmd: list[np.ndarray] = []
        self._servo_meas: list[np.ndarray] = []
        self.events: list[dict] = []      # connect / start / stall / stop / error ... (sidecar JSON)
        self.meta: dict = {}              # driver.describe() etc. (sidecar JSON)

    def __len__(self) -> int:
        return len(self._t)

    def append(self, t: float, cmd, meas, stop: bool = False, *, fresh: bool = True, dt: float = float("nan"),
               servo_cmd=None, servo_meas=None) -> None:
        nan = np.full(self.n, np.nan)
        # diagnostics first, t last: a reader on another thread sizes its snapshot by len(t)
        self._fresh.append(int(bool(fresh)))
        self._dt.append(float(dt))
        self._servo_cmd.append(nan if servo_cmd is None else np.asarray(servo_cmd, float).reshape(self.n).copy())
        self._servo_meas.append(nan if servo_meas is None else np.asarray(servo_meas, float).reshape(self.n).copy())
        self._stop.append(int(bool(stop)))
        self._cmd.append(np.asarray(cmd, float).reshape(self.n).copy())
        self._meas.append(np.asarray(meas, float).reshape(self.n).copy())
        self._t.append(float(t))

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t, cmd, meas) of the rows complete so far -- safe while another thread appends."""
        k = len(self._t)
        return (np.asarray(self._t[:k], float), np.asarray(self._cmd[:k], float).reshape(-1, self.n),
                np.asarray(self._meas[:k], float).reshape(-1, self.n))

    @property
    def fresh(self) -> np.ndarray:
        return np.asarray(self._fresh, int)

    @property
    def dt(self) -> np.ndarray:
        return np.asarray(self._dt, float)

    @property
    def servo_cmd(self) -> np.ndarray:
        return np.asarray(self._servo_cmd, float).reshape(-1, self.n)

    @property
    def servo_meas(self) -> np.ndarray:
        return np.asarray(self._servo_meas, float).reshape(-1, self.n)

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self._t, float)

    @property
    def cmd(self) -> np.ndarray:
        return np.asarray(self._cmd, float).reshape(-1, self.n)

    @property
    def meas(self) -> np.ndarray:
        return np.asarray(self._meas, float).reshape(-1, self.n)

    @property
    def stop(self) -> np.ndarray:
        return np.asarray(self._stop, int)

    @property
    def stop_time(self) -> float | None:
        s = self.stop
        return float(self.t[int(np.argmax(s))]) if s.any() else None

    def as_deg(self) -> dict:
        """The dict ``hwcompare.read_log`` returns (deg), for ``metrics.hw_report``."""
        return dict(t=self.t, cmd=np.degrees(self.cmd), meas=np.degrees(self.meas),
                    stop=self.stop.astype(float) if self.stop.any() else None)

    def save(self, path, fmt: str = "361") -> None:
        if fmt == "361":
            write_log_361(path, self.t, np.degrees(self.cmd), np.degrees(self.meas), self.stop)
        elif fmt == "461":
            write_log_461(path, self.t, self.cmd, self.meas)
        else:
            raise ValueError(f"unknown log format {fmt!r}")

    def save_sidecars(self, path) -> tuple[Path, Path]:
        """Write ``<path stem>.meta.json`` and ``<path stem>.diag.csv`` next to the log ``path``."""
        path = Path(path)
        meta_p = path.with_name(path.stem + ".meta.json")
        diag_p = path.with_name(path.stem + ".diag.csv")
        fresh = self.fresh
        summary = dict(rows=len(self), stale_rows=int((fresh == 0).sum()),
                       stalls=sum(e.get("kind") == "stall" for e in self.events),
                       max_dt=float(np.nanmax(self.dt)) if len(self) and np.isfinite(self.dt).any() else None,
                       stop_time=self.stop_time)
        meta_p.write_text(json.dumps(dict(meta=self.meta, summary=summary, events=self.events, log=path.name),
                                     indent=2, default=_json_default))
        n = self.n
        header = (["t", "dt", "fresh", "stop"] + [f"servo_cmd_{j + 1}" for j in range(n)]
                  + [f"servo_meas_{j + 1}" for j in range(n)])
        fmt = lambda x: "" if not np.isfinite(x) else f"{x:.4f}"
        with open(diag_p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for k in range(len(self)):
                w.writerow([f"{self._t[k]:.4f}", fmt(self._dt[k]), self._fresh[k], self._stop[k]]
                           + [fmt(x) for x in self._servo_cmd[k]] + [fmt(x) for x in self._servo_meas[k]])
        return meta_p, diag_p


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
