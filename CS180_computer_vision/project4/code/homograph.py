import cv2
import numpy as np
import matplotlib.pyplot as plt

from hg_helper import *


class RANSAC:
    
    def __init__(self, num_pts_per_sample = 4, percentage_of_outliers = 0.6 ,threshold = 3):
        self.num_pts_per_sample = num_pts_per_sample
        self.percentage_of_outliers = percentage_of_outliers
        self.threthold = threshold
        self.iter_times = np.round(np.log(1 - 0.99) / 
                                   np.log(1 - (1 - self.percentage_of_outliers) ** self.num_pts_per_sample))
    
    def random_choose_n_points(self, n, num_of_points):
        
        # num_of_points are num of points in total
        # n is num of points we choose
        
        chosen_index = np.random.choice(num_of_points, n, replace=False)
        all_points_index = np.arange(num_of_points)
        left_points_index = np.setdiff1d(all_points_index, chosen_index)
        
        return chosen_index, left_points_index
    
    def get_in_out_liner_pts(self, h, match_pairs, threshold):
        
        inputs = match_pairs[:, 0:2]
        outputs = match_pairs[:, 2:4]
        
        input_pts = add_1_for_homograph(inputs)
        trans_pts = np.matmul(h, input_pts.T)
        trans_pts = (trans_pts / trans_pts[-1, :]).T
        # now trans_pts: [x1, y1, 1]
        
        distance = np.linalg.norm(trans_pts[:, 0:2] - outputs, axis=1)
        
        inlier_index = np.where(distance <= threshold)
        outliner_index = np.where(distance > threshold)
        
        inliners = match_pairs[inlier_index]
        outliners = match_pairs[outliner_index]
        
        return inliners, outliners
        
    def choose_best_h(self, correspondence, kind, name, save_mid = False):
        
        num_of_points = correspondence.shape[0]
        num_pts_per_sample = self.num_pts_per_sample
        
        cur_inlier_pts = []
        cur_outlier_pts = []
        cur_chosen_pts = []
        cur_inlier_num = 0
        
        percentage_of_inliers = 1 - self.percentage_of_outliers
        Min_inlier_num = percentage_of_inliers * num_of_points
        
        # record the process
        inliners_num_list = []
        outliners_num_list = []
        iter_list = []
        
        iters = 0
        
        while iters <= self.iter_times:
            
            #print(f"in iter{iters}") 
            
            chosen_index, left_points_index = self.random_choose_n_points(num_pts_per_sample ,num_of_points)
            chosen_pts, left_pts = correspondence[chosen_index], correspondence[left_points_index]
            
            inputs, outputs = chosen_pts[:, 0:2], chosen_pts[:, 2:4]
            temp_H = solve_homograpy(inputs, outputs)
            
            # now we evaluate the temp H
            threthold = self.threthold
            inliners, outliners = self.get_in_out_liner_pts(temp_H, left_pts, threthold)
            num_of_inliers = inliners.shape[0]
            
            if(num_of_inliers > Min_inlier_num):
                #print(f"num_of_inlier{num_of_inliers} > Min_inlier_num")
                if(num_of_inliers > cur_inlier_num):
                    #print(f"num_of_inlier{num_of_inliers} more than {cur_inlier_num}, better!")
                    cur_inlier_pts = inliners
                    cur_inlier_num = num_of_inliers
                    cur_outlier_pts = outliners
                    cur_chosen_pts = chosen_pts
                    
                    totol_num = cur_inlier_num + cur_chosen_pts.shape[0]
                    outliners_num = outliners.shape[0]
                    inliners_num_list.append(totol_num)
                    outliners_num_list.append(outliners_num)
                    iter_list.append(iters)
                    
                else:
                    pass
                    #print("not a better result")
            else:
                pass
                #print("less than Min_inlier_num")
                  
            iters += 1
            
        best_match_pairs = np.concatenate((cur_chosen_pts, cur_inlier_pts), axis=0)
        #print(best_match_pairs[:5])
        inputs =  best_match_pairs[:, 0:2]
        outputs = best_match_pairs[:, 2:4]
        best_H = solve_homograpy(inputs, outputs)
        params = [cur_inlier_num, cur_inlier_pts, cur_outlier_pts, cur_chosen_pts]
        
        print(inliners_num_list)
        print(outliners_num_list)
        if(save_mid):
            plt.plot(iter_list, inliners_num_list, color='green', label='Inliners')
            plt.plot(iter_list, outliners_num_list, color='red', label='Outliners')
            plt.legend()
            plt.title(f"{name} : Inliners vs Outliners")
            plt.xlabel('Iterations')
            plt.ylabel('Number of Points')
            save_path = "./../mid_result/" + kind + "/" + name + "_RANSAC.jpg"
            plt.savefig(save_path)
            plt.close()
        
        return params, best_H
    
    def show_best(self, chosen_pairs, img1, img2, save_path):
        
        img = np.hstack((img1, img2))
        h, w, _ = img1.shape
        
        for corrd in chosen_pairs:
            x1, y1, x2, y2 = corrd
            
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2) + w)
            y2 = int(round(y2))
            
            cv2.circle(img, (x1,y1), radius=5,  color=(0, 0, 255), 
                       thickness = 2, lineType=cv2.LINE_AA)
            cv2.circle(img, (x2,y2), radius=5,  color=(0, 0, 255), 
                       thickness = 2, lineType=cv2.LINE_AA)
            cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0),
                     thickness=2)
            
        cv2.imwrite(save_path, img)