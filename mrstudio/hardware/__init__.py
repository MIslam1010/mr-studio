"""Hardware bridge for the course arm (myCobot-280 class 6R): drivers, safety checklist, logging and
the hardware acceptance tests of the SBEG 361/461 project hardware sessions.

    base       ArmSpec, COURSE_ARM, ArmDriver interface, course_arm_robot()
    simulated  SimulatedArm (first-order servo lag + quantisation + noise), ManualClock
    mycobot    MyCobotDriver (pymycobot, imported lazily)
    safety     course-arm hardware safety checklist (SafetySupervisor)
    logger     CSV logs in the 361 hwcompare.py / 461 hardware.py formats
    metrics    HW-2 ... HW-6 (as hwcompare.hw_report); validation_checks()
    session    quintic moves (MR 9.2.2.1) and the GUI-independent run loop
"""
from .base import COURSE_ARM, COURSE_ARM_NAME, ArmDriver, ArmSpec, HardwareError, course_arm_robot  # noqa: F401
from .simulated import ManualClock, SimulatedArm  # noqa: F401
