Project1
Name: Yuanteng Chen
Cal ID: 3039725444

!!!Since gradescope's commit file size is limited to 100MB, I can't upload base_line and SSD results, so only Gradient results are in the result folder and index.html. After putting raw images into "data/" and running the command: python main.py --algorithm SSD --run_mode auto --base_line True
you can get all results and the web page can display the results of the first two columns normally.!!!

1.file structure:

submission.zip
 |-README.txt: Basical info and running command
 |-index.html: Code explanation and result presentation
 |-data: Ground truth of cathedral.jpg
 |-result: Including the result of baseline(directly merged without align), result of SSD(using 
	SSD metric to align) and result of Gradient(using Gradient metric to align)

 |-util.py: functions of preprocessing raw images(split, crop, merge ...), functions of finding
	best align and functions of Image Pyramid.
 |-main.py: define all the parameters and using the functions of util.py to finish the alignment.

 |-other_images: other images used in index.html (you can ignore it)


2.running command:

!!!first you should put all the raw images(includeing .jpg and .tif) into "data/" so that main.py 
   can find these raw images(to avoid submitting files that were too large, I deleted these raw images)!!!

command1: test on all raw images
(1)align all raw images in "data/*" using SSD metric:
python main.py --algorithm SSD --run_mode auto --base_line True

(2)align all raw images in "data/*" using NCC metric:
python main.py --algorithm NCC --run_mode auto --base_line True

(3)align all raw images in "data/*" using Gradient metric:
python main.py --algorithm Gradient --run_mode auto --base_line True

(if you want to save the result of baseline(merged directly), you can set --base_line True)

command2: test on one specific raw images:
(1)test SSD metric alignment on cathedral.jpg only and evaluate the result
python main.py --algorithm SSD --run_mode single --path data/cathedral.jpg --evaluate True 
(to test other metric you can set --algorithm NCC/Gradient)

command3: test on one specific raw images and adjust window_size, crop rate and levels of pyramid:
python main.py --algorithm SSD --run_mode single --path data/cathedral.jpg --evaluate True \
	       --window_size 32 --level 4 --crop_rate 0.03

The defination of these hyperparameters can be checked in main.py

3.Code explanation:
you can see concrete explanation of code in index.html.
