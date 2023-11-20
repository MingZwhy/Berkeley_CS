Project4_part2
Name: Yuanteng Chen
Cal ID: 3039725444


1.file structure:

submission.zip
 |-code: all source code of the project （code.ipynb show the processing of debug）
 |-README.txt: Basical info and file structure
 |-index.html: Code explanation and result presentation
 |-data: including all raw images
 |-mid-result: Including mid results of the project
 |-final-result: Including mosaic results of the project

run command:

cd code
python mosaic.py --data_dir_path 5pic_a --algorithm MOP --if_save_mid_result True --if_crop_rawimages True

args:
--data_dir_path = 3pic / 4pic / 5pic_a / 5pic_b
--algorithm = MOP / SIFT
--if_save_mid_result = True / False
--if_crop_rawimages = True / False


You can adjust other params not in args including:
specific crop_size: (1008,756) in default as too high resolution will take too much time.
MOP_params: (nip=500, threshold=0.75) in default.

2.Code explanation:
you can see concrete explanation of code in index.html.
