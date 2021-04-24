import time
import brainflow
import numpy as np
import threading

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

from scipy import signal

from skimage.transform import resize

from tensorflow import keras

# ######################################### This should be a different file
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
# #########################################################################


class DataThread(threading.Thread):

    def __init__(self, board, board_id):
        threading.Thread.__init__(self)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.keep_alive = True
        self.board = board

    def run(self):
        # Initialize Acquisition Paramenters
        window_size = 3
        sleep_time = 0.5
        points_per_update = window_size * self.sampling_rate
        # Load Decoder
        classifier = keras.models.load_model('..\jupyter-notebooks\classifier_22_v3')
        # Initialize Actuator
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

        _ = input()

        while self.keep_alive:
            start_time = time.time()
            time.sleep(sleep_time)
            # get current board data doesnt remove data from the buffer
            data = self.board.get_current_board_data(int(points_per_update))
            # print('Data Shape %s' % (str(data.shape)))
            first_channel = True
            for channel in self.eeg_channels:
                # filters work in-place
                DataFilter.perform_highpass(data[channel], self.sampling_rate, 0.5, 4, FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandstop(data[channel], self.sampling_rate, 60.0, 2.0, 4, FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandpass(data[channel], self.sampling_rate, 19.0, 12.0, 4, FilterTypes.BUTTERWORTH.value, 0)

                stft_data = data[channel].copy()
                _, _, Zxx = signal.stft(stft_data, self.sampling_rate, nperseg=128, noverlap=127, nfft=256)
                spectro = np.asarray(Zxx[7:32, :])
                spectro = np.abs(spectro)
                spectro = np.expand_dims(spectro, axis=2)

                if first_channel:
                    tfi = spectro.copy()
                    first_channel = False
                else:
                    tfi = np.dstack((tfi, spectro))

            res_tfi = resize(tfi, (64, 64, 8))
            res_tfi = np.reshape(res_tfi, (1, 64, 64, 8))
            outputs = classifier.predict(res_tfi)
            outputs = np.reshape(outputs, -1)
            output = np.argmax(outputs)
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
            print(time.time()-start_time)

        disable_motor(MOTOR_ID_Y, packetHandler, portHandler)
        disable_motor(3, packetHandler, portHandler)
        disable_motor(2, packetHandler, portHandler)
        disable_motor(MOTOR_ID_Z, packetHandler, portHandler)

        # Close the port
        portHandler.closePort()


def main():
    BoardShim.enable_dev_board_logger()

    # use synthetic board for demo
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    data_thread = DataThread(board, board_id)
    data_thread.start()
    try:
        time.sleep(120)
    finally:
        data_thread.keep_alive = False
        data_thread.join()

    board.stop_stream()
    board.release_session()


if __name__ == "__main__":
    main()
