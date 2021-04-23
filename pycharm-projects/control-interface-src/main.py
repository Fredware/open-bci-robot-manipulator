import numpy as np
from robot_config import *
from dynamixel_sdk import *


def enable_motor(motor_id, packetHandler, portHandler):
    # Enable the motor so that it can start being controlled Takes in the portHandler, id of the motor to control,
    # address of the torque enable, and the value of 1 to enable it
    motor_result, motor_error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADD_XL_TORQUE_ENABLE,
                                                             TORQUE_ENABLE)
    if motor_result != COMM_SUCCESS:  # This series of conditional statements is just making sure everything is good
        print("%s" % packetHandler.getTxRxResult(motor_result))
    elif motor_error != 0:
        print("%s" % packetHandler.getRxPacketError(motor_error))
    else:
        print('Motor was successfully enabled')


def disable_motor(motor_id, packetHandler, portHandler):
    # Disable the motor
    motor_result, motor_error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADD_XL_TORQUE_ENABLE,
                                                             TORQUE_DISABLE)
    if motor_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(motor_result))
    elif motor_error != 0:
        print("%s" % packetHandler.getRxPacketError(motor_error))


def move_up(position, packetHandler, portHandler):
    position = position - POS_INCREMENT
    if position > 3000:
        position = 3000
    move_motor(position, MOTOR_ID_Y, packetHandler, portHandler)
    return position


def move_down(position, packetHandler, portHandler):
    position = position + POS_INCREMENT
    if position < 800:
        position = 800
    move_motor(position, MOTOR_ID_Y, packetHandler, portHandler)
    return position


def move_right(position, packetHandler, portHandler):
    position = position + POS_INCREMENT
    if position > 3000:
        position = 3000
    move_motor(position, MOTOR_ID_Z, packetHandler, portHandler)
    return position


def move_left(position, packetHandler, portHandler):
    position = position - POS_INCREMENT
    if position < 800:
        position = 800
    move_motor(position, MOTOR_ID_Z, packetHandler, portHandler)
    return position


def move_motor(position, motor_id, packetHandler, portHandler):
    # Write the goal position to the motor
    motor_result, motor_error = packetHandler.write4ByteTxRx(portHandler, motor_id, ADD_XL_GOAL_POSITION,
                                                             position)
    if motor_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(motor_result))
    elif motor_error != 0:
        print("%s" % packetHandler.getRxPacketError(motor_error))

    # Continue checking the current position until you reach the goal
    while 1:
        motor_position, motor_result, motor_error = packetHandler.read4ByteTxRx(portHandler, motor_id,
                                                                                ADD_XL_PRESENT_POSITION)
        if motor_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(motor_result))
        elif motor_error != 0:
            print("%s" % packetHandler.getRxPacketError(motor_error))

        print('[ID: %03d] GoalPos:%03d PresPos:%03d' % (motor_id, position, motor_position))

        if not abs(position - motor_position) > POS_THRESHOLD:
            break


def main():
    # LIF Parameters:
    alpha = 0.4
    beta = 0.1
    v_ref = 0
    v_thresh = 1

    v_prev = np.zeros(4)
    v = np.zeros(4)

    # Manipulator Initialization
    portHandler = PortHandler(DEVICE_NAME)

    # There's also a class type that handles how to send packages across the COM port
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open the port
    if portHandler.openPort():
        print('Succeeded to open the port')
    else:
        print('Failed to open the port')
        quit()

    enable_motor(MOTOR_ID_Y, packetHandler, portHandler)
    enable_motor(3, packetHandler, portHandler)
    enable_motor(2, packetHandler, portHandler)
    enable_motor(MOTOR_ID_Z, packetHandler, portHandler)

    position_my = 1900
    position_mz = 1900

    move_motor(position_my, MOTOR_ID_Y, packetHandler, portHandler)
    move_motor(1900, 3, packetHandler, portHandler)
    move_motor(1900, 2, packetHandler, portHandler)
    move_motor(position_mz, MOTOR_ID_Z, packetHandler, portHandler)

    loop_ok = True

    while loop_ok:
        #     raw = get_data()
        #     filtered = filter_data(raw)
        #     tfi = transform_data(filtered)
        #     output = classify_data()
        output = int(input())

        exc = np.zeros(4)

        if output == 0:
            exc[0] = 1
        elif output == 2:
            exc[1] = 1
        elif output == 3:
            exc[2] = 1
        elif output == 4:
            exc[3] = 1

        print(v)
        dv = np.subtract((alpha * exc), (beta * v_prev))
        v = np.add(v_prev, dv)
        print(v)

        if v[0] > v_thresh:
            position_my = move_down(position_my, packetHandler, portHandler)
            v[0] = v_ref
        elif v[1] > v_thresh:
            position_mz = move_left(position_mz, packetHandler, portHandler)
            v[1] = v_ref
        elif v[2] > v_thresh:
            position_mz = move_right(position_mz, packetHandler, portHandler)
            v[2] = v_ref
        elif v[3] > v_thresh:
            position_my = move_up(position_my, packetHandler, portHandler)
            v[3] = v_ref
        else:
            print('do not move')

        v_prev = v

        print('=' * 79)
        if output == 5:
            loop_ok = False

    disable_motor(MOTOR_ID_Y, packetHandler, portHandler)
    disable_motor(3, packetHandler, portHandler)
    disable_motor(2, packetHandler, portHandler)
    disable_motor(MOTOR_ID_Z, packetHandler, portHandler)

    # Close the port
    portHandler.closePort()


if __name__ == "__main__":
    main()
