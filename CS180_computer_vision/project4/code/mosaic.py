import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

from homograph import *
from get_points import *
from trans_image import *

class Mosaic:
    
    def __init__(self, kind = "2pic", algo = "MOP", MOP_params = [500, 0.75] , crop_size=None, save_mid = False):
        self.dir_path = "./../data/" + kind + "/"
        self.save_dir_path = "./../final_result/" + kind + "/"
        self.images = []

        files = os.listdir(self.dir_path)
        mid_index = int(len(files) / 2)
        self.standard = mid_index
        standard = files[mid_index]
        self.ex = os.path.splitext(standard)[1]

        for file_name in os.listdir(self.dir_path):
            image = cv2.imread(os.path.join(self.dir_path + file_name))
            if(crop_size != None):
                image = cv2.resize(image, crop_size)
            self.images.append(image)
                
        example = self.images[self.standard]
        self.height = example.shape[0]
        self.width = example.shape[1]
        self.nip = MOP_params[0]
        self.threshold = MOP_params[1]
        self.algo = algo
        self.kind = kind
        self.save_mid = save_mid
        
    def trans_all_imgs(self):
        
        # initialize the H
        Hs = {}
        
        num_imgs = len(self.images)
        
        for i in range(num_imgs - 1):
            name = f"H{i}{i+1}"
            
            src_img = self.images[i]
            dst_img = self.images[i+1]
            
            if(algo == "SIFT"):
                matched_pairs_finder = SIFT_points(src_img, dst_img, 
                                        num_features=2000, threshold=0.6, save_path=None)
                match_pairs = matched_pairs_finder.get_points_and_show()
                match_pairs = np.array(match_pairs)
            elif(algo == "MOP"):
                matched_pairs_finder = MOP_points(src_img, dst_img, str(i), str(i+1),
                                                  kind, self.nip, self.threshold, self.save_mid)
                match_pairs = matched_pairs_finder.get_points()
                print("match_pairs: ", len(match_pairs))
                match_pairs = np.array(match_pairs)
            
            best_H_finder = RANSAC()
            params, best_h = best_H_finder.choose_best_h(match_pairs, kind, name = f"{i}_{i+1}", save_mid=self.save_mid)
            #inlier_pts, outlier_pts, chosen_pts = params[1:]
            #print(best_h)
            
            Hs[name] = best_h
            
        right_Hs = self.get_right_H(Hs)
        print(right_Hs)
        self.jigsaw(right_Hs)
            
            
        
    def get_right_H(self, Hs):
        
        print(Hs.keys())
        
        num_imgs = len(Hs) + 1
        
        # as we don't need to wrap mid pic
        # set H for mid pic to E
        
        mid_name = f"H{self.standard}{self.standard}"
        Hs[mid_name] = np.eye(3)
        
        mid_index = self.standard
        
        for i in range(0, mid_index):
            name = f"H{i}{mid_index}"
            begin = np.eye(3)
            for j in range(i, mid_index):
                H_name = f"H{j}{j+1}"
                H = Hs[H_name]
                begin = np.matmul(H, begin)
                
            Hs[name] = begin 
            
        for i in range(mid_index+1, num_imgs):
            name = f"H{i}{mid_index}"
            begin = np.eye(3)
            
            j = i - 1
            
            while j >= mid_index:
                H_name = f"H{j}{j+1}"
                H = Hs[H_name]
                H_inv = np.linalg.inv(H)
                begin = np.matmul(H_inv, begin)
                
                j -= 1
                
            Hs[name] = begin 
            
        return Hs
                
    def extent(self, H, width, height):
        
        origin_corners = np.array([[0,0],[width,0],[width,height],[0,height]])
        
        corners_pts = np.concatenate((origin_corners, np.ones((4,1))), axis=1)
        estimate_corners = np.matmul(H, corners_pts.T)
        estimate_corners = estimate_corners / estimate_corners[-1, :]
        
        min_p = np.amin(estimate_corners.T, axis=0)
        max_p = np.amax(estimate_corners.T, axis=0)
        
        return min_p, max_p
                
    
    def paint_whole_canvas(self, Hs, height, width):
        
        # find the max height and width after trans
        
        min_canvas = np.array([np.inf, np.inf, np.inf])
        max_canvas = np.array([-np.inf, -np.inf, -np.inf])
        
        num_images = len(self.images)
        
        for i in range(num_images):
            name = f"H{i}{self.standard}"
            H = Hs[name]
            
            min_p, max_p = self.extent(H, width, height)
            print(f"{name} : min_p: {min_p}, max_p : {max_p}")
            min_canvas = np.minimum(min_canvas, min_p)
            max_canvas = np.maximum(max_canvas, max_p)
            
        distance = np.ceil(max_canvas - min_canvas)
            
        extent_canvas_width = int(distance[0] + 1)
        extent_canvas_height = int(distance[1] + 1)
        
        # paint the whole canvas with max height and width
        
        channels = 3
        canvas = np.zeros((extent_canvas_height,extent_canvas_width,channels), dtype=np.int64)
        mask = np.ones((extent_canvas_height,extent_canvas_width))
        
        offset = min_canvas.astype(np.int64)
        print(f"canvas size: ({extent_canvas_height},{extent_canvas_width})")
        print("offset is: ")
        print(offset)
        offset[2] = 0
        
        return canvas, mask, offset
         
    def jigsaw(self, Hs):
        canvas, mask, offset = self.paint_whole_canvas(Hs, self.height, self.width)

        for i, img in enumerate(self.images):
            
            name = f"H{i}{self.standard}"
            H = Hs[name]
            H_inv = np.linalg.inv(H)
            
            src_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            canvas = trans_img_into_target(src_img, canvas, mask, H_inv, offset)
            
            # put areas of trans_canvas in mask to 0
            areas = np.where(canvas)[:2]
            mask[areas] = 0
            #plt.imshow(mask)
            #plt.show()
            
            save_path = self.save_dir_path + '/' + self.algo + "_" + str(i) + self.ex
            cv2.imwrite(save_path, canvas[:,:,(2,1,0)])
            
            
if __name__ == "__main__":
    
    desc = "Choose the dir_path of data(raw images), algorithm used to get matched points pairs" \
           "whether save the mid result, crop size of raw images and hyperparams of MOP"
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-d", "--data_dir_path", required=False, default="5pic_a",
        help="5pic_a / 5pic_b / 3pic / 4pic"
    )
    parser.add_argument(
        "-l", "--algorithm", required=False, default="MOP",
        help="choose the algorithm to get points, MOP or SIFT"
    )
    parser.add_argument(
        "-m", "--if_save_mid_result", required=False, default=True,
        help="whether or not to save middle result of whole processing"
    )
    parser.add_argument(
        "-c", "--if_crop_rawimages", required=False, default=True,
        help="if true, we will crop raw images into (1008, 756)"
    )
    args = parser.parse_args()
    
    kind = args.data_dir_path
    algo = args.algorithm
    save_mid = args.if_save_mid_result
    crop_size = (1008, 756) if args.if_crop_rawimages else None
    MOP_params = [500, 0.75]
    
    mosaicer = Mosaic(kind, algo, MOP_params, crop_size, save_mid)
    mosaicer.trans_all_imgs()