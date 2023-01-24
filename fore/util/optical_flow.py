import cv2 as cv
import numpy as np


def calc_optical_flow(video):
    """
    :param video: video
    :type video:np.array
    :return: optical flow as an array of len(video)-1
    :rtype: np.array
    """
    img_array_new = np.zeros_like(video, dtype=np.uint8)
    size = (video.shape[1], video.shape[2])
    for i in range(len(video)):
        new_image = np.zeros((video[i].shape[0], video[i].shape[1], 3), dtype=np.uint8)
        new_image[:, :, 0] = video[i][:, :, 2]
        new_image[:, :, 1] = video[i][:, :, 1]
        new_image[:, :, 2] = video[i][:, :, 0]
        img_array_new[i] = new_image

    first_frame = img_array_new[0]

    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally
    # expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255
    out_optical = np.zeros((len(img_array_new) - 1, *size, 3), dtype=np.uint8)
    for i in range(1, len(img_array_new)):
        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        frame = img_array_new[i]
        # Opens a new window and displays the input
        # frame
        # cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                           None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        # from shape (channel , width , height) to (width , height , channel)
        # rgb = rgb.transpose((2, 1, 0))
        # Flip image to back to rgb
        flip_rgb = np.zeros_like(rgb, dtype=np.uint8)
        flip_rgb[:, :, 0] = rgb[:, :, 2]
        flip_rgb[:, :, 1] = rgb[:, :, 1]
        flip_rgb[:, :, 2] = rgb[:, :, 0]

        out_optical[i - 1] = flip_rgb

        # Updates previous frame
        prev_gray = gray
    return out_optical
