import matplotlib.pyplot as plt
from align_image_code import align_images
import cv2
import numpy as np
from skimage import io, color


def Lowpass_Filter(image, low_cutoff):
    # using guassian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), low_cutoff)
    return blurred_image


def Impulse_Filter(image):
    # using
    impulsed_image = cv2.Laplacian(image, cv2.CV_64F)
    return impulsed_image


def Highpass_Filter(image, low_cutoff, mode=1):
    # mode 1: highpass = origin - lowpass
    # mode 2: highpass = Impulse_Filter(origin) - lowpass

    blurred_image = cv2.GaussianBlur(image, (5, 5), low_cutoff)

    if (mode == 2):
        impulsed_image = Impulse_Filter(image)
        highpass = cv2.subtract(impulsed_image, blurred_image)
        return highpass
    elif (mode == 1):
        highpass = cv2.subtract(image, blurred_image)
        sharpen_image = cv2.add(image, highpass)
        return highpass
    else:
        print("wrong mode: ", mode)
        return None


def hybrid_image(low_image, high_image, freq_forlow, freq_forhigh, mode=1):
    # blurred_image
    blurred_image = Lowpass_Filter(low_image, freq_forlow)

    # sharpen_image
    sharpen_image = Highpass_Filter(high_image, freq_forhigh, mode)

    # hybrid_image
    hybrid_image = cv2.add(blurred_image, sharpen_image)

    return hybrid_image


# First load images

# high sf
im1 = plt.imread('data/Trump1.jpg')

# low sf
im2 = plt.imread('data/Biden1.jpg')

# Next align images (this code is provided, but may be improved)
im2_aligned, im1_aligned = align_images(im2, im1)

#print(im1_aligned.shape, im2_aligned.shape)
io.imsave('data/aligned_Trump.jpg', im1_aligned)
io.imsave('data/aligned_Biden.jpg', im2_aligned)
