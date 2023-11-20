import cv2
import numpy as np
from skimage.feature import corner_harris, peak_local_max
import glob
import matplotlib.pyplot as plt

class Mouse_points:
    
    def __init__(self, img1_path, img2_path, save_dir_path, size = (756,504)):
        self.img1_path = img1_path
        self.img2_path = img2_path
            
        self.save_dir_path = save_dir_path
        self.size = size

    def mouse_callback(self, event, x, y, flags, params):
        images, points1, points2 = params[0], params[1], params[2]
        
        height, width, _ = images[0].shape
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # record position
            points1.append((x, y))
            print("Left image: Clicked at (x={}, y={})".format(x, y))
            cv2.circle(images[0], (x, y), 2, (0, 255, 0), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # display = cv2.hconcat(images)
            # as we hconcat two images, the x in right image 
            # should be x in whole image - width of left image
            x = x - width
            points2.append((x, y))
            print("Right image: Clicked at (x={}, y={})".format(x, y))
            cv2.circle(images[1], (x, y), 2, (0, 255, 0), -1)

    def capture_points(self):
        points1 = []
        points2 = []
        
        image1 = cv2.imread(self.img1_path)
        image1 = cv2.resize(image1, self.size)
        image2 = cv2.imread(self.img2_path)
        image2 = cv2.resize(image2, self.size)
                
        images = [image1, image2]

        cv2.namedWindow("Images")
        params = [images, points1, points2]
        cv2.setMouseCallback("Images", self.mouse_callback, params)

        print("Begin")

        while True:
            # show two images
            display = cv2.hconcat(images)
            cv2.imshow("Images", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                if self.save_dir_path is not None:
                    for i, image in enumerate(images):
                        path = self.save_dir_path + "/" + "mouse_points_" + str(i) + ".jpg"
                        cv2.imwrite(path, images[i])
                break

        cv2.destroyAllWindows()
        
        points_pair = []
        for i in range(len(points1)):
            point = [points1[0],points1[1],points2[0],points2[1]]
            points_pair.append(point)
            
        point_path = self.save_dir_path + "/mouse_points.npy"
        np.save(point_path, points_pair)
        
        return points_pair
    
class MOP_points:
    
    def __init__(self, img1, img2, name1, name2, kind="2pic", nip=500, threshold=0.75, save_mid = False):
            
        self.image1 = img1
        self.image2 = img2
        self.img1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
            
        self.nip = nip
        self.threshold = threshold
        self.save_mid = save_mid
        self.kind = kind
        
        self.name1 = name1
        self.name2 = name2
        
    def show_all_coords(self, coords1, coords2, desc = "_all"):
        show1 = self.image1.copy()
        show2 = self.image2.copy()
        
        for i in range(coords1.shape[1]):
            x, y = coords1[0][i], coords1[1][i]
            
            cv2.circle(show1, (y, x), radius=1, color=[0,255,0],
                thickness=2, lineType=cv2.LINE_AA)
            
        for i in range(coords2.shape[1]):
            x, y = coords2[0][i], coords2[1][i]
            
            cv2.circle(show2, (y, x), radius=1, color=[0,255,0],
                thickness=2, lineType=cv2.LINE_AA)
            
        show1_path = "./../mid_result/" + self.kind + "/" + self.name1 + desc + "_coords.jpg"
        show2_path = "./../mid_result/" + self.kind + "/" + self.name2 + desc + "_coords.jpg"
        
        cv2.imwrite(show1_path, show1)
        cv2.imwrite(show2_path, show2)
        
    def show_selected_coords(self, coords1, coords2, desc = "_selected"):
        show1 = self.image1.copy()
        show2 = self.image2.copy()
        
        for i in range(len(coords1)):
            x, y = coords1[i][0], coords1[i][1]
            
            cv2.circle(show1, (y, x), radius=1, color=[0,255,0],
                thickness=2, lineType=cv2.LINE_AA)
            
        for i in range(len(coords2)):
            x, y = coords2[i][0], coords2[i][1]
            
            cv2.circle(show2, (y, x), radius=1, color=[0,255,0],
                thickness=2, lineType=cv2.LINE_AA)
            
        show1_path = "./../mid_result/" + self.kind + "/" + self.name1 + desc + "_coords.jpg"
        show2_path = "./../mid_result/" + self.kind + "/" + self.name2 + desc + "_coords.jpg"
        
        cv2.imwrite(show1_path, show1)
        cv2.imwrite(show2_path, show2)
        
    def show_match(self, match_pairs):
        image = np.hstack((self.image1, self.image2))
        height, width, _ = self.image1.shape
        
        for pair in match_pairs:
            x1,y1 = pair[0], pair[1]
            x2,y2 = pair[2], pair[3]
                    
            x2 += width
                    
            x1,y1,x2,y2 = round(x1),round(y1),round(x2),round(y2)
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    
            cv2.circle(image, (x1, y1), radius=3, color=[255,0,0],
                        thickness=2, lineType=cv2.LINE_AA)

            cv2.circle(image, (x2, y2), radius=3, color=[255,0,0],
                        thickness=2, lineType=cv2.LINE_AA)

            cv2.line(image, (x1, y1), (x2, y2), color=[255,255,0],
                        thickness=1)
            
        save_path = "./../mid_result/" + self.kind + "/" + self.name1 + "_" + self.name2 + "_pairs.jpg"
        cv2.imwrite(save_path, image)
    
    def get_harris_corners(self, im, edge_discard=20):
        """
        This function takes a b&w image and an optional amount to discard
        on the edge (default is 5 pixels), and finds all harris corners
        in the image. Harris corners near the edge are discarded and the
        coordinates of the remaining corners are returned. A 2d array (h)
        containing the h value of every pixel is also returned.

        h is the same shape as the original image, im.
        coords is 2 x n (ys, xs).
        """

        assert edge_discard >= 20

        # find harris corners
        h = corner_harris(im, method='eps', sigma=1)
        coords = peak_local_max(h, min_distance=1, indices=True)

        # discard points on edge
        edge = edge_discard  # pixels
        mask = (coords[:, 0] > edge) & \
            (coords[:, 0] < im.shape[0] - edge) & \
            (coords[:, 1] > edge) & \
            (coords[:, 1] < im.shape[1] - edge)
        coords = coords[mask].T
        return h, coords
        
    def anms(self, h, coords, nip = 500, robust = 0.9, l_r = "left"):
        # Get the intensity of corners
        corners_h = []
        num_of_corners = coords.shape[1]
        for i in range(num_of_corners):
            h_index = coords[0][i]
            w_index = coords[1][i]
            corner_h = h[h_index][w_index]
            corners_h.append(corner_h)
            
        # Get the indices of corners sorted in descending order of corner response
        corners_h = np.array(corners_h)
        idx = np.argsort(-corners_h)
        #print(idx)

        # Initialize the list of selected corner coordinates
        selected_coords = []
        selected_h = []

        # Set the initial suppression radius to infinity
        suppression_radius = np.inf
        
        # record the process
        iter_list = []
        num_coords_list = []
        sup_radius_list = []

        # Perform Adaptive Non-Maximal Suppression
        for i in range(num_of_corners):
            current_coord = np.array((coords[0][idx[i]], coords[1][idx[i]]))
            current_h = corners_h[idx[i]]

            # Calculate the minimum suppression radius for the current interest point
            min_radius = np.inf
            
            if(len(selected_coords) == 0):
                selected_coords.append(current_coord)
                selected_h.append(current_h)
                continue

            for j in range(len(selected_coords)):
                dist = np.linalg.norm(current_coord - selected_coords[j])
                if corners_h[idx[i]] < robust * selected_h[j]:
                    min_radius = min(min_radius, dist)

            # If the suppression radius of the current interest point is greater than or equal 
            # to the previously retained interest points' suppression radius, add it to the selected corner list
            if min_radius >= robust * suppression_radius:
                selected_coords.append(current_coord)
                selected_h.append(current_h)
                
            # Update the suppression radius
            suppression_radius = min(min_radius, suppression_radius)
            #print(suppression_radius)
            
            iter_list.append(i)
            num_coords_list.append(len(selected_coords))
            sup_radius_list.append(suppression_radius)

            # Stop selection if the desired number of interest points is reached
            if len(selected_coords) >= nip:
                break
            
        if(self.save_mid):
            plt.plot(iter_list, num_coords_list, color='green', label='selected_coords')
            plt.plot(iter_list, sup_radius_list, color='red', label='radius')
            plt.legend()
            plt.title(f"{self.name1}_{self.name2} : coords suppression")
            plt.xlabel('Iterations')
            plt.ylabel('Number of Points')
            if(l_r == "left"):
                save_path = "./../mid_result/" + self.kind + "/" + self.name1 + "_ANMS.jpg"
            else:
                save_path = "./../mid_result/" + self.kind + "/" + self.name2 + "_ANMS.jpg"
            plt.savefig(save_path)
            plt.close()

        return selected_coords
    
    def normalize_descriptor(self, descriptor):
        mean = np.mean(descriptor)
        std = np.std(descriptor)
        normalized_descriptor = (descriptor - mean) / std
        return normalized_descriptor

    def extract_descriptor(self, image, coords, window_size = 40, patch_size = 8, scale = 5):
        descriptors = []
        half_wsize = window_size // 2

        for coord in coords:
            y, x = coord[0], coord[1]
            patch = image[y - half_wsize: y + half_wsize, x - half_wsize: x + half_wsize]  # get the window
            sampled_patch = patch[::scale, ::scale]
            normalized_patch = self.normalize_descriptor(sampled_patch)  # normalize
            descriptor = normalized_patch.flatten()
            descriptors.append(descriptor)
            
        #print(descriptors[0])
        return descriptors
    
    def feature_match(self, desc1, desc2, coords1, coords2, threshold = 0.72):
        
        match_pairs = []
        
        for i in range(len(desc1)):
            pt1 = coords1[i]
            dc1 = desc1[i]
            
            distance = np.linalg.norm(dc1 - desc2, axis=1)
            idx = np.argsort(distance)
            best_index = idx[0]
            next_index = idx[1]
            best_dis = distance[best_index]
            next_dis = distance[next_index]
            
            rate = best_dis / next_dis
            #print(rate)
            if (rate < threshold):
                pt2 = coords2[best_index]
                match_pair = [pt1[1],pt1[0],pt2[1],pt2[0]]
                match_pairs.append(match_pair)
                
        return match_pairs
    
    def get_points(self):
        h1, coords1 = self.get_harris_corners(self.img1)
        h2, coords2 = self.get_harris_corners(self.img2)
        
        if(self.save_mid):
            self.show_all_coords(coords1, coords2, "_all")
        
        selected_coords1 = self.anms(h1, coords1, self.nip, 0.9, "left")
        selected_coords2 = self.anms(h2, coords2, self.nip, 0.9, "right")
        
        if(self.save_mid):
            self.show_selected_coords(selected_coords1, selected_coords2, "_selected")
        
        descriptors1 = self.extract_descriptor(self.img1, selected_coords1)
        descriptors2 = self.extract_descriptor(self.img2, selected_coords2)
        
        match_pairs = self.feature_match(descriptors1, descriptors2, selected_coords1, selected_coords2, self.threshold)
        
        if(self.save_mid):
            self.show_match(match_pairs)
        
        return match_pairs

class SIFT_points:
    
    def __init__(self, img1, img2, num_features, threshold, save_path):
            
        self.img1 = img1
        self.img2 = img2
            
        self.num_features = num_features
        self.threshold = threshold
        self.save_path = save_path
        
    def get_points(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = cv2.xfeatures2d.SIFT_create(self.num_features)
        keypoints, descriptors = features.detectAndCompute(gray, None)
        
        keypoints_list = [point.pt for point in keypoints]
        return (keypoints_list, descriptors)
    
    def valid_points_pair(self, k_d_1, k_d_2):
        keypoints1, descriptor1 = k_d_1
        keypoints2, descriptor2 = k_d_2

        valid_points_pair = []
        num_pairs = len(keypoints1)
        
        for i in range(num_pairs):
            likeness = np.linalg.norm(descriptor1[i] - descriptor2, axis=1)
            likeness_sorted_index = np.argsort(likeness)
            
            most_like_feature, next_like_feature \
                = likeness[likeness_sorted_index[0]], likeness[likeness_sorted_index[1]]
                
            if (most_like_feature / next_like_feature) < self.threshold:
                point1 = keypoints1[i]
                point2 = keypoints2[likeness_sorted_index[0]]
                point_pair = [point1[0], point1[1], point2[0], point2[1]]
                valid_points_pair.append(point_pair)
                
        print(f"there are {len(valid_points_pair)} valid matched pairs in total")
        return valid_points_pair
    
    def save_features_result(self, valid_points_pair, save_path):
        
        image = np.hstack((self.img1, self.img2))

        height, width, _ = self.img1.shape

        for pair in valid_points_pair:
            x1,y1 = pair[0], pair[1]
            x2,y2 = pair[2], pair[3]
            
            # same as situation in get_points
            # we concat two images in width
            # so x2 in image should be x2 in image2 + width
            
            x2 += width
            
            x1,y1,x2,y2 = round(x1),round(y1),round(x2),round(y2)
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            
            cv2.circle(image, (x1, y1), radius=3, color=[255,0,0],
                        thickness=2, lineType=cv2.LINE_AA)

            cv2.circle(image, (x2, y2), radius=3, color=[255,0,0],
                        thickness=2, lineType=cv2.LINE_AA)

            cv2.line(image, (x1, y1), (x2, y2), color=[255,255,0],
                        thickness=1)
            
        #cv2.imshow("sift_features", image)
        #cv2.waitKey(0)

        cv2.imwrite(save_path, image)
        
    def get_points_and_show(self, if_save=False):
        k_d_1 = self.get_points(self.img1)
        k_d_2 = self.get_points(self.img2)
        
        valid_points_pair = self.valid_points_pair(k_d_1, k_d_2)
        if(if_save):
            self.save_features_result(valid_points_pair, self.save_path)
            
        return valid_points_pair
            

if __name__ == "__main__":
    dir_path = './data/2pic/'
    images = glob.glob(dir_path + "*")
    print(images)