import cv2
import numpy as np
import copy

from skimage.metrics import structural_similarity as compare_ssim

'''
split the raw image into (B,G,R)
'''

def split_raw_pic(raw_pic, crop_rate):
    raw_height, width, channel = raw_pic.shape
    height = raw_height // 3

    #B,G,R from top to bottom
    blue3 = raw_pic[:height, : , : ]
    green3 = raw_pic[height:2*height, : , : ]
    red3 = raw_pic[2*height:3*height, : , : ]

    blue = blue3[:,:,0]
    green = green3[:,:,0]
    red = red3[:,:,0]

    croped_blue = crop_pic(blue, crop_rate)
    croped_green = crop_pic(green, crop_rate)
    croped_red = crop_pic(red, crop_rate)

    height, width = croped_blue.shape

    return (height,width), croped_blue, croped_green, croped_red

def crop_pic(raw_pic, rate=0.025):
    '''
    cropping image at the rate
    '''
    height,width = raw_pic.shape[0], raw_pic.shape[1]
    crop_h = int(np.floor(height * rate))
    crop_w = int(np.floor(width * rate))

    return raw_pic[crop_h : height - crop_h, crop_w : width - crop_w]

'''
define the displacement functions
'''

def trans_pic(img, h, w):
    new_pic = np.roll(img, (h,w), (0,1))
    return new_pic

'''
merge three channels into color image
'''

def merge_into_color(blue, green, red):
    return cv2.merge([blue, green, red])

'''
define the metric we will use:
1: Sum of Squared Differences (SSD)
2: Normalized Cross-Correlation (NCC)
3: gradient of images
'''

def SSD_metric(img1, img2):
    '''

    :param img1: always be Blue channel
    :param img2: Green channel or Red channel
    :return: SSD of img1 and img2
    '''

    return np.sum((img1 - img2) ** 2)

def NCC_metric(img1, img2):
    '''

    :param img1: always be Blue channel
    :param img2: Green channel or Red channel
    :return: NCC of img1 and img2

    img1:(H,W)
    img2:(H,W)
    '''

    correlation = np.sum(img1 * img2)
    normalization = np.sqrt(np.sum(img1**2) * np.sum(img2**2))
    ncc = correlation / normalization

    return ncc


def Gradients_metric(img1, img2):

    #print(img1.shape,img2.shape)
    # 计算梯度图像
    gradient_img1 = cv2.Sobel(img1, cv2.CV_32FC1, 1, 1, ksize=3)
    gradient_img2 = cv2.Sobel(img2, cv2.CV_32FC1, 1, 1, ksize=3)

    # 计算梯度差异
    diff = cv2.absdiff(gradient_img1, gradient_img2)

    # 计算平均差异值
    mean_diff = np.mean(diff)

    return mean_diff


'''
try best align in window
'''

def find_best_align(img1, img2, window_size=15, center=(0,0), metric=1):
    '''

    :param img1: always be Blue channel
    :param img2: Green channel or Red channel
    :param window_size: try to find best align in (window_size,window_size)
    :param metric: how to evaluate the result
            1: SSD
            2: NCC
    :return: displacement vector (x,y)
    '''

    best_displacement = [0,0]
    best_score = -1000000
    #best_align = img2

    #try different displacement and find the best
    #in NCC and gradients, actually higher ncc or gradients, worse performance,
    #in order to use the same best score, we use the inverse of ncc or gradients

    for i in range(-window_size+center[0],window_size+center[0]):
        for j in range(-window_size+center[1],window_size+center[1]):

            new_img2 = trans_pic(img2, i, j)

            if(metric==1):
                score = -SSD_metric(img1,new_img2)
            elif(metric==2):
                score = NCC_metric(img1,new_img2)
            elif(metric==3):
                score = -Gradients_metric(img1,new_img2)

            if(score > best_score):
                #find better displacement and update the score and vector
                best_score = score
                best_displacement[0] = i
                best_displacement[1] = j
                #best_align = new_img2

    #print("best score: ", best_score)
    #print("best shift: ", best_displacement)

    best_align = trans_pic(img2, best_displacement[0], best_displacement[1])

    return best_align, best_displacement

def image_pyramid(img1, img2, window_size=15, level = 3, metric=1):
    #param level: level of pyramid

    c_img1 = copy.deepcopy(img1)
    c_img2 = copy.deepcopy(img2)

    origin_size = c_img1.shape
    best_displacement = (0,0)

    for i in range(level):
        rate = 2 ** (level-i-1)
        w_size = window_size // (2 ** i)
        if(rate == 1):
            d_img1 = c_img1
            d_img2 = c_img2
        else:
            d_img1 = cv2.resize(c_img1, (origin_size[1] // rate, origin_size[0] // rate))
            d_img2 = cv2.resize(c_img2, (origin_size[1] // rate, origin_size[0] // rate))

        _, best_displacement = find_best_align(d_img1, d_img2, \
            window_size=w_size, center=best_displacement, metric=metric)
        #print("in level ", i, " align: ", best_displacement)

    #print("final align: ", best_displacement)
    best_align = trans_pic(c_img2, best_displacement[0], best_displacement[1])

    return best_align, best_displacement

'''
evaluate the result:
the metric is psnr,
calculate the psnr between our result with ground_truth provided by website
'''

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    max_pixel = np.max(img1)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    return compare_ssim(img1, img2, multichannel=True)