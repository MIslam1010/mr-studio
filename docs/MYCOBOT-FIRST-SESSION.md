# myCobot 280: protocol for the first real-arm session

*MR Studio, Hardware tab. Print this page and tick each box as you go. Written 2 November 2026 for pymycobot 4.0.7.*

This is the first time MR Studio drives the real arm. Until now the driver has only run against software:
a fake SDK in `tests/test_hardware.py`, and the real SDK talking to a firmware emulator in
`tests/test_mycobot_emulator.py`. So go slowly. Every step says what to do, what you should see, and when to
stop. If you see anything this page does not describe, stop and write it down before you go on.

**People.** At least two:
* the **operator** sits at the laptop, outside the taped zone, and presses **STOP** (or **Esc**);
* the **power person** keeps a hand on the arm's power switch for the whole session, watches the arm (not the
  screen), and switches off without waiting to be told when anything looks wrong.

**Two ways to stop.** **STOP / Esc** in MR Studio halts the arm and *latches*: the arm accepts no further
commands until you press **Reset latch**. The **power switch** cuts the servos. The arm then goes limp and
can drop, so use it when the software stop has not worked or something is clearly wrong.

**Abort rule (applies to every step).** Press STOP **and** switch the power off if:
* the arm moves when it should not, moves the wrong joint, or turns the wrong way;
* it vibrates, buzzes loudly, jerks, smells hot, or hits anything;
* MR Studio shows **STOP NOT CONFIRMED**, freezes for more than a second, or reports a *stall* (or stops with
  "streaming stalled");
* anyone steps inside the taped zone while the arm is powered.

After an abort, do not reset and repeat the step until you know why it happened and have written it down.

---

## A. The day before (at your desk, no arm)

Open a Terminal in the project folder: `cd "$HOME/Documents/Edu/Lectures/Robotics/MR Studio Project/MR-Studio"`.

| ☐ | Do | You should see | Stop if |
|---|---|---|---|
| ☐ A1 | `.venv/bin/python -m pip install -r requirements-hardware.txt` (already done on this Mac on 2 Nov 2026) | pymycobot 4.0.7 installed, or "Requirement already satisfied" | pip fails |
| ☐ A2 | `.venv/bin/python tools/mycobot_dryrun.py` (dry run: opens no port and moves nothing) | `all 7 SDK methods present`, six `OK` lines for the servo clamps, `Result: OK`. Two warnings are normal: no arm plugged in, and no calibration file yet | any `FAIL` line |
| ☐ A3 | `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_mycobot_emulator.py tests/test_hardware.py` | all pass, about 60 s for the emulator file | any failure |
| ☐ A4 | **Rehearsal with the emulator.** Terminal 1: `.venv/bin/python tools/mycobot_emulator.py -q` and note the printed port (`/dev/ttys…`). Terminal 2: `MRSTUDIO_MYCOBOT_CALIBRATION=/tmp/rehearsal_cal.json .venv/bin/python run_app.py`. Open the Hardware tab, choose **myCobot 280 (USB serial)**, type the emulator's port, **Connect**, save a calibration, tick the checklist, run a small move, press STOP / Esc, **Reset latch**, then close the app. | Connect works, and so do Run, STOP (status "arm halted and latched") and Reset latch. Terminal 1 prints the frame counts when you press Ctrl-C | anything fails. **The variable keeps the rehearsal calibration out of the real file**, so never save a rehearsal calibration without it |
| ☐ A5 | Print this page and the course safety checklist. Make sure that every person who will touch the arm has completed the safety induction (rule 5). | | |

## B. Bench setup (arm unplugged, power off)

| ☐ | Do | You should see | Stop if |
|---|---|---|---|
| ☐ B1 | Clamp or bolt the arm's base to the table. Clear the table. Tape a keep-out circle of radius **0.58 m** around the base (reach + 0.30 m, rule 2). | The base does not move when you push the arm firmly by hand | the base can tip or slide |
| ☐ B2 | Put the laptop and the operator **outside** the circle. Put the power switch (or the power supply's plug) where the power person can reach it without entering the circle. | | the power person cannot reach the switch from outside the circle |
| ☐ B3 | Route the USB cable with some slack, and make sure you can pull it out by hand from outside the circle (you need this in step E2). | | |
| ☐ B4 | Put the arm in a low, folded rest pose, with a soft pad under the tool. | If it went limp now, it would not fall far | |
| ☐ B5 | Weigh anything that is attached to the tool flange. | ≤ 0.125 kg (rule 4) | heavier: remove it |

## C. Power, port and read-only checks (no motion)

| ☐ | Do | You should see | Stop if |
|---|---|---|---|
| ☐ C1 | Switch on the arm's power. Put the controller in **Transponder** mode (its screen menu; see the arm's manual). Plug the USB cable into the Mac. | The arm stays still | the arm moves |
| ☐ C2 | `.venv/bin/python tools/mycobot_dryrun.py` | Section 4 shows a `/dev/cu.…` port marked **myCobot USB bridge (CP210x / CH340 / CH9102)** and the line "likely arm port: …" | no such port: check the cable and the driver (CH340/CH9102 boards may need the WCH driver) |
| ☐ C3 | `.venv/bin/python tools/mycobot_dryrun.py --probe /dev/cu.<port>`. This opens the port and sends **read commands only** (joint angles and power state), never motion. | "reads are fast enough for the 50 Hz stream", a median of a few ms, 0 failed reads, and six servo angles that look like the arm's pose. **Save this output** (`… > probe.txt`). A median of 20–80 ms gives the warning "slow reads … the stream will alternate set-point and read ticks": MR Studio then writes a set-point on one tick and reads the joints on the next, so each tick waits for one round trip (about 7–10 set-points/s at 40–60 ms, tested with the emulator: no stall, and the 2 s test move still ends in about 2.6 s). That is acceptable for the slow H1 moves of this session; write the median down | the arm moves. All reads fail: try `--baud 1000000`, and check Transponder mode. Some reads fail, or "reads too slow to stream" (median above 80 ms, or any read above 100 ms): do not stream. Write the numbers down and stop here |

## D. Connect, stop test and calibration (no commanded motion)

| ☐ | Do | You should see | Stop if |
|---|---|---|---|
| ☐ D1 | Start MR Studio (`.venv/bin/python run_app.py`), open the **Hardware** tab, choose **myCobot 280 (USB serial)**, pick the port, baud **115200**, and click **Connect**. Connect switches the servos on. | "connected: myCobot (serial) on /dev/cu.… at 115200 baud" and a warning that there is **no calibration file**. The servos stiffen and hold the pose (a click or a hum is normal). The health line reads "last fresh … ms ago · 0 failed reads" | the arm moves more than a degree or two. Connect fails twice: check the baud rate. Connect fails with "did not acknowledge" or "fresh mode": the firmware does not answer the way pymycobot 4.0.7 expects (Connect also switches on the firmware's *fresh mode*, so it never queues set-points). Note the message and the firmware version, and do not continue |
| ☐ D2 | **Software stop test:** press **STOP** (or Esc) while the arm is idle, then **Reset latch**. | "STOP — arm halted and latched", then "Latch reset — torque on" | a red **STOP NOT CONFIRMED** |
| ☐ D3 | **Power switch test (rule 7):** the power person switches off and then on again, while the operator watches the screen. Then **Disconnect** and **Connect** again. | The arm goes limp onto its pad. MR Studio reports failed reads or a communication error, which is expected. After you reconnect, the readings are fresh again | the arm drops off the pad: improve the rest pose (B4) first |
| ☐ D4 | **Calibration (commissioning).** For each joint you need its *sign* (does the servo count up when the model angle increases?) and its *offset* (the servo angle when the model angle is 0). Use the definition $q_{servo} = s\,q_{model} + o$. Choose a reference pose that you can recognise both on the real arm and in the 3-D view (for example, a link vertical, checked with a set square). Read the servo angles from C3 (or from the log's `servo_meas` columns), work out $o = q_{servo} - s\,q_{model}$ on paper, then enter the signs (**invert**) and the offsets in **myCobot calibration** and click **Save calibration**. | "Saved ~/.mrstudio/mycobot_calibration.json". The 3-D view now shows the same pose as the real arm, joint by joint, to within a few degrees | the view and the arm still disagree by more than about 5° on any joint: fix that joint's offset or sign before you go on. A sign you are not sure of is checked in E1 |
| ☐ D5 | Copy the calibration file next to your session notes, with today's date in the file name. Tick the checklist in the tab: induction, keep-out, clear, e-stop tested (D2 + D3), supervisor, payload. Choose session **H1** (≤ 25 % speed). | The **Run** button becomes active only after a calibration file exists and every box is ticked | |

## E. First motions (session H1: at most 25 % of rated speed, 40 °/s per joint)

Use **Move to the joint target**, so the tab plans a smooth quintic move. Save the log after every run with **Save log…**:
it writes the CSV, a `.meta.json` file (port, SDK version, calibration, events) and a `.diag.csv` file (raw servo angles, timing).

| ☐ | Do | You should see | Stop if |
|---|---|---|---|
| ☐ E1 | **One joint, +10°.** Set the target to the current pose plus **10° on J1** (the base: it carries no gravity load). Click **Run**. Then run back to the start pose. | A smooth turn of about 10° in roughly a second, **in the same direction as the 3-D view**, and the status "Run complete". The report shows no tracking-error stop and no stalls. The measured curve follows the commanded one within a few degrees | the joint turns the other way (STOP, tick **invert** for J1, Save, repeat E1); "tracking error" (the firmware streaming % is too low for this speed: raise **Streaming speed (firmware)** by 10 %, Save calibration, repeat); any stall. **The first Run ends at once with "send_angles: the arm on … did not acknowledge the command (pymycobot returned -1)"**: the firmware does not acknowledge SEND_ANGLES (pymycobot 4.0.7 waits for a reply to every command). The driver stops and latches on purpose, because it treats a missing acknowledgement as a lost set-point; the arm should not have moved. Note the message and the firmware version and **do not continue** (do not retry or work around it): the driver must first be checked against that firmware |
| ☐ E2 | **Stop during motion (HW-6).** Repeat E1 with **Stop test (HW-6) at t = 0.5 s**. | The arm stops part-way and the status says "Stopped … arm LATCHED". The HW-6 verdicts in the report pass. **Reset latch** | it keeps moving after the stop |
| ☐ E3 | **Cable-pull test.** Start E1 again, slowly. During the move, pull the USB cable out (from outside the circle). | MR Studio ends the run at once (the emulator test shows within one 20 ms tick; on a slow link, within one round trip) with **STOP NOT CONFIRMED — switch off the arm's power now**. That message is expected here: the halt command cannot reach an unplugged arm. **The power person switches off.** The arm should already have stopped within a degree or two of where it was | the arm keeps moving after the pull, more than a few degrees. That means the controller is working through queued commands although Connect set fresh mode: note it and do not continue until it has been looked into |
| ☐ E4 | Power on, plug the cable back in, **Disconnect / Connect** (the port name can change: pick it again), and **Reset latch**. | Fresh readings, and the pose matches the 3-D view | the pose no longer matches: recalibrate (D4) |
| ☐ E5 | Repeat E1 for each other joint, one at a time, ±10°: J2, J3 and J4 carry gravity load, so watch the measured curve closely. | Each joint behaves like J1, in the right direction, with no stop or stall | as in E1 |
| ☐ E6 | **Two joints together:** J1 +10° and J2 +10° in one move. | A smooth combined move, "Run complete" | as in E1 |
| ☐ E7 | **The course trajectory:** **Commanded joints of a CSV log**, then **Load CSV…** and choose the course's trajectory. Leave session H1 (25 %), because this is the first run of a new motion. The tab slows the trajectory down if it is too fast. Keep the HW-6 stop test **off** for this run. | An approach move to the start pose, then the trajectory, then "Run complete". The report's HW tests pass | any tracking-error stop, any stall, or an HW verdict that fails |

**If the controller stops answering** while the port stays open (a hung controller), MR Studio latches the
run within about **0.4 s** (the emulator test measures 0.41 s, also with 50–60 ms round trips) and shows **STOP NOT CONFIRMED**, because the
halt cannot be acknowledged: switch the power off, as for any unconfirmed stop. pymycobot 4.0.7 itself keeps
waiting about 1.5 s on the hung call; MR Studio stops waiting after 0.4 s, and its halt command goes out once
the SDK gives up. If every tick takes longer than 100 ms for a whole second (a link too slow to stream), MR
Studio stops and latches the run with "streaming stalled" instead of holding the arm on a frozen trajectory:
treat it like a failed C3 probe. **If MR Studio freezes during a run** (no screen updates, STOP seems to do nothing), switch
the power off: that is not expected and must be written down.

## F. End of the session

| ☐ | Do | You should see |
|---|---|---|
| ☐ F1 | Run a move back to the rest pose (B4). Then press STOP to latch. | The arm is in its rest pose and latched |
| ☐ F2 | Hold the arm with one hand. Click **Torque off** and confirm. | The arm goes limp **and drops**: let it down onto its pad |
| ☐ F3 | **Disconnect**, switch the power off, unplug the USB cable. | |

## Keep these files (one folder per session, with the date in its name)

* `probe.txt` (C3), and the dry-run output (`tools/mycobot_dryrun.py > dryrun.txt`).
* The calibration JSON (D5).
* For every run: the `.csv` log, `.meta.json` and `.diag.csv`.
* A photo of the bench setup, and this sheet with the boxes ticked and notes on anything unexpected.

| Step | Time | OK? | Notes |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
