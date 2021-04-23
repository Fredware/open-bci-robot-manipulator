import numpy as np

# Control table addresses for the XL series motors (servo motors in the PincherX 100)
ADD_XL_TORQUE_ENABLE = 64
ADD_XL_GOAL_POSITION = 116
ADD_XL_PRESENT_POSITION = 132

# Protocol version is 2.0 for the XL series motors
PROTOCOL_VERSION: float = 2.0

# Default settings
MOTOR_ID = 4  # I think this is the identifier for which motor to control (I guess there should be 5 of these 1-5 maybe)
BAUD_RATE = 57600
DEVICE_NAME = 'COM4'  # It's a com port for windows machines

# Motor Limits
# [ID, lower, upper]
MOTOR_LIMITS = np.array([[1, 0, 4095], [2, 800, 3000], [3, 800, 3000], [4, 800, 3200], [5, 1600, 2800]])

TORQUE_ENABLE = 1  # Values to enable and disable the torque on the motors (turn them on/off)
TORQUE_DISABLE = 0
POS_THRESHOLD = 20  # Threshold of the difference between our current position and desired position
GOAL_POSITION = 3200

POS_INCREMENT = 250

MOTOR_ID_Y = 4
MOTOR_ID_Z = 1
