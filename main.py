import cv2
import numpy as np
import time


class Config:
    MIN_RANGE = 0
    MAX_RANGE = 40
    DEBUG = True

def get_contours(frame):

    kernel = np.ones((3, 3), np.uint8)

    frame = cv2.inRange(frame, Config.MIN_RANGE, Config.MAX_RANGE)
    frame = cv2.erode(frame, kernel, iterations=5)
    frame = cv2.dilate(frame, kernel, iterations=9)

    # https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
    countours, _ = cv2.findContours(
        frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Debug frame
    if Config.DEBUG:
        cv2.imshow(f'Eroded / Dilated', frame)

    return countours


def calculate_distance(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    borders = get_contours(gray_frame)
    x, y, w, h = cv2.boundingRect(borders[0])

    frame_height, frame_width, _ = frame.shape

    line_middle = (x + w // 2, y + h // 2)
    frame_middle = (frame_width // 2, frame_height // 2)
    max_distance = frame_width // 2

    distance_percentage = (
        100 * (line_middle[0] - frame_middle[0])) // max_distance

    if Config.DEBUG:
        pass

    return distance_percentage


def main():
    capture = cv2.VideoCapture("vid.mp4")

    while (frame := capture.read()[1]) is not None:
        distance_percentage = calculate_distance(frame)

        # Do stuff with distance
        print(distance_percentage)

        cv2.imshow('Main', frame)
        cv2.waitKey(1)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
