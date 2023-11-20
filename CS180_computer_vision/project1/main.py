import argparse
import cv2
import numpy as np
import glob
import time

import utils

if __name__ == '__main__':
    # Parse command line arguments
    desc = "Choose the algorithm, window_size, levels of pyramid and run mode"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-a", "--algorithm", required=True, default="SSD",
        help="the algorithm to process pic, SSD / NCC / Gradient"
    )
    parser.add_argument(
        "-c", "--crop_rate", required=False, default=0.025,
        help="the crop rate of raw images"
    )
    parser.add_argument(
        "-w", "--window_size", required=False, default=32,
        help="the window size of the base of the pyramid"
    )
    parser.add_argument(
        "-l", "--level", required=False, default=4,
        help="the level of image pyramid"
    )
    parser.add_argument(
        "-b", "--base_line", required=False, default=False,
        help="whether to save base_line (merge directly without alignment)"
    )
    parser.add_argument(
        "-r", "--run_mode", required=True, default="auto",
        help="auto--process all images ; single--process single image"
    )
    parser.add_argument(
        "-p", "--path", required=False, default="data/emir.tif",
        help="when running in single mode, must provive right path of image"
    )
    parser.add_argument(
        "-e", "--evaluate", required=False, default=False,
        help="Only evaluate the result on cathedral.jpg"
    )

    args = parser.parse_args()

    dir_path = "data/"
    save_dir_path = "result/"

    sota_path = "data/cathedral_gt.jpg"
    if(args.evaluate):
        gt_image = cv2.imread(sota_path)

    algorithm = 1   # default SSD
    if(args.algorithm == 'SSD'):
        algorithm = 1
    elif(args.algorithm == 'NCC'):
        algorithm = 2
    elif(args.algorithm == 'Gradient'):
        algorithm = 3
    else:
        print('wrong algorithm choice')
        exit(1)

    crop_rate = args.crop_rate
    window_size = args.window_size
    level = args.level
    mode = args.run_mode
    # if processing jpg, no need to use image pyramid
    pyramid = False

    if(mode == 'auto'):
        raw_images = glob.glob(dir_path + '*')
    elif(mode == 'single'):
        raw_images = [args.path]

    for raw_image in raw_images:
        raw_image = raw_image.replace('\\', '/')
        name = raw_image.split('/')[-1].split('.')[0]
        if(name == "cathedral_gt"):
            continue
        print("processing ", raw_image)
        if(raw_image.split('.')[-1] == 'tif'):
            image = cv2.imread(raw_image, 1)
            pyramid = True
        elif(raw_image.split('.')[-1] == 'jpg'):
            image = cv2.imread(raw_image)
            pyramid = False
        else:
            image = cv2.imread(raw_image)
            pyramid = False

        # change the type and turn into (0,1)
        image = image.astype(np.float32) / 255.0

        # split raw image
        (height, width), blue, green, red = utils.split_raw_pic(image, crop_rate)
        # merge directly -- base_line
        base_line = cv2.merge([blue, green, red])
        base_line = base_line * 255.0

        if(args.base_line):
            path = save_dir_path + name + "_baseline.jpg"
            cv2.imwrite(path, base_line, [cv2.IMWRITE_JPEG_QUALITY, 100])

        start_time = time.time()

        if(pyramid):
            green_align, green_vector = utils.image_pyramid(blue, green, window_size, level, algorithm)
            red_align, red_vector = utils.image_pyramid(blue, red, window_size, level, algorithm)
        else:
            green_align, green_vector = utils.find_best_align(blue,green,window_size,(0,0),algorithm)
            red_align, red_vector = utils.find_best_align(blue,red,window_size,(0,0),algorithm)

        end_time = time.time()
        using_time = end_time-start_time
        print("using time: {:.2f} s".format(using_time))

        best_align = utils.merge_into_color(blue, green_align, red_align)
        best_align = best_align * 255.0
        path = save_dir_path + name + "_" + args.algorithm + "_align.jpg"
        print("green shift vector: ", green_vector)
        print("red shift vector: ", red_vector)
        print("save result to ", path)
        print()
        cv2.imwrite(path, best_align, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # evaluate the result
        if(args.evaluate):
            gt_image = cv2.resize(gt_image, (width,height))
            psnr = utils.calculate_psnr(gt_image, best_align)
            print("Metric: ", args.algorithm, "; psnr: ", psnr)