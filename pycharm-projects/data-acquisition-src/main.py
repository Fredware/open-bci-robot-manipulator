import argparse
import numpy as np
import cv2 as cv
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter


# Functions for generating visual cues:
def make_arrow(img, head, body):
    rgb_color = (200, 200, 200)
    body_pts = body.reshape((-1, 1, 2))
    cv.fillPoly(img, pts=[body_pts], color=rgb_color)

    head_pts = head.reshape((-1, 1, 2))
    cv.fillPoly(img, pts=[head_pts], color=rgb_color)

    make_cross(img, 8)


def make_cross(img, thickness):
    rgb_color = (150, 150, 150)
    cv.line(img, (960, 450), (960, 510), rgb_color, thickness)
    cv.line(img, (930, 480), (990, 480), rgb_color, thickness)


def rotate(pts, center, deg):
    angle = np.deg2rad(deg)
    rot_mat = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    o = np.atleast_2d(center)
    p = np.atleast_2d(pts)
    return np.squeeze((rot_mat @ (p.T - o.T) + o.T).T)


def main():
    # =========================================== Boiler Plate Code
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    # =========================================== Experiment Setup
    # Change the resolution based on your machine: (height x width x channels)
    window_size = (960, 1920, 3)

    # Define the points that form the polygons of the arrow
    up_body_pts = [(870, 350), (870, 900), (1050, 900), (1050, 350)]
    up_head_pts = [(960, 50), (745, 350), (1175, 350)]

    # (x,y) coordinates of the center of the screen
    center = (960, 480)

    # Make an image of an arrow pointing up
    up_arrow = np.zeros(window_size, np.uint8)
    body = np.array(up_body_pts, np.int32)
    head = np.array(up_head_pts, np.int32)
    make_arrow(up_arrow, head, body)

    # Make an image of an arrow pointing down
    down_arrow = np.zeros(window_size, np.uint8)
    body = np.array(rotate(up_body_pts, center, 180), np.int32)
    head = np.array(rotate(up_head_pts, center, 180), np.int32)
    make_arrow(down_arrow, head, body)

    # Make an image of an arrow pointing left
    left_arrow = np.zeros(window_size, np.uint8)
    body = np.array(rotate(up_body_pts, center, 270), np.int32)
    head = np.array(rotate(up_head_pts, center, 270), np.int32)
    make_arrow(left_arrow, head, body)

    # Make an image of an arrow pointing right
    right_arrow = np.zeros(window_size, np.uint8)
    body = np.array(rotate(up_body_pts, center, 90), np.int32)
    head = np.array(rotate(up_head_pts, center, 90), np.int32)
    make_arrow(right_arrow, head, body)

    # Make a blank screen with black background
    blank_screen = np.zeros(window_size, np.uint8)

    # Make an image of a fixation cross on a black background
    blank_cross = np.copy(blank_screen)
    make_cross(blank_cross, 10)

    # Organize all images in a list
    visual_cues = [blank_screen, up_arrow, down_arrow, left_arrow, right_arrow]

    # Generate a random sequence of 48 visual cues ensuring that the cues are uniformly distributed:
    rand_seq = [1, 2, 3, 4]
    rand_seq = np.repeat(rand_seq, 4)
    rand_seq = np.random.permutation(rand_seq)

    # Show a blank screen and immediately move the window to the upper left corner so that it occupies the entire screen
    cv.imshow('window', blank_screen)
    cv.moveWindow('window', 0, 0)
    cv.waitKey(1)

    # Make a board object and let the API initialize the board
    board = BoardShim(args.board_id, params)
    board.prepare_session()

    # 50 trials/session * 8 sec/trial * 250 samples/sec = 100 000 * SF = 1.25 = 125000 samples in buffer
    board.start_stream(125000, args.streamer_params)

    # =========================================== Running the Experiment
    for code in rand_seq:
        # Display blank screen for 2 seconds
        board.insert_marker(6)
        cv.imshow('window', blank_screen)
        cv.waitKey(2000)

        # Display blank screen with fixation cross for 2 seconds
        board.insert_marker(5)
        cv.imshow('window', blank_cross)
        cv.waitKey(2000)

        # Display visual cue with fixation cross for 1.25 seconds followed by fixation cross for 2.75 seconds
        board.insert_marker(code)
        cv.imshow('window', visual_cues[code])
        cv.waitKey(1250)
        cv.imshow('window', blank_cross)
        cv.waitKey(2750)

    cv.destroyAllWindows()

    # Get all the data stored in the buffer and terminate all background processes
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    # Save data to the current directory as a .csv file
    # DataFilter.write_file(data, 'ses_14_run_03.csv', 'w')

# Procedure before recording session:
# [ ] Disconnect computer from main line.
# [ ] Put computer on airplane mode
# [ ] Put nearby devices on airplane mode
# [ ] Turn off lights if possible
# [ ] Verify impedance values in OpenBCI GUI
# [ ] Practice once before starting the session
# [ ] Change name of log file


if __name__ == "__main__":
    main()
