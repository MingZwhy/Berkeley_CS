import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def trans_img_into_target(src_img, dst_img, mask, h, offset = np.array([0, 0, 0])):
    
    y, x = np.where(mask)
    dst_pts = np.concatenate((x[:,np.newaxis], y[:, np.newaxis], np.ones((x.size, 1))), axis=1)
    dst_pts = dst_pts + offset
    
    out_src = np.matmul(h, dst_pts.T)
    out_src = out_src / out_src[-1,:]
    out_src = out_src[0:2, :].T
    dst_pts = dst_pts[:, 0:2].astype(np.int64)
    
    h, w, _ = src_img.shape
    
    left_top = np.floor(out_src[:, ::-1])
    left_top = left_top.astype(np.int64)
    right_bottom = np.ceil(out_src[:, ::-1])
    right_bottom = right_bottom.astype(np.int64)
    
    dst_pts = dst_pts - offset[:2]
    
    valid_mask = np.logical_and.reduce([
        ~np.logical_or(np.any(left_top < 0, axis=1), np.any(right_bottom < 0, axis=1)),
        ~np.logical_or(left_top[:, 0] >= h-1, left_top[:, 1] >= w-1),
        ~np.logical_or(right_bottom[:, 0] >= h-1, right_bottom[:, 1] >= w-1)
    ])
    
    dst_pts = dst_pts[valid_mask]
    out_src = out_src[valid_mask]
    left_top = left_top[valid_mask]
    right_bottom = right_bottom[valid_mask]
    
    right_top = np.concatenate((left_top[:, 0:1], right_bottom[:, 1:2]), axis=1)
    left_bottom = np.concatenate((right_bottom[:, 0:1], left_top[:, 1:2]), axis=1)
    
    num = out_src.shape[0]
    
    weights = np.zeros((num, 4))
    weights[:,0] = np.linalg.norm(left_top - out_src[:, ::-1], axis=1)
    weights[:,1] = np.linalg.norm(right_top - out_src[:, ::-1], axis=1)
    weights[:,2] = np.linalg.norm(left_bottom - out_src[:, ::-1], axis=1)
    weights[:,3] = np.linalg.norm(right_bottom - out_src[:, ::-1], axis=1)
    
    weights[np.all(weights == 0, axis=1)] = 1
    weights = 1 / weights
    
    weighted_lt = src_img[left_top[:,0], left_top[:,1], :] * weights[:, 0:1]
    weighted_rt = src_img[right_top[:,0], right_top[:,1], :] * weights[:, 1:2]
    weighted_lb = src_img[left_bottom[:,0], left_bottom[:,1], :] * weights[:, 2:3]
    weighted_rb = src_img[right_bottom[:,0], right_bottom[:,1], :] * weights[:, 3:4]
    
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weighted_value = \
        (weighted_lt+weighted_rt+weighted_lb+weighted_rb) / weights_sum
        
    changed_dst_image = dst_img.copy()
    changed_dst_image[dst_pts[:,1], dst_pts[:,0],:] = weighted_value
    
    return changed_dst_image
    