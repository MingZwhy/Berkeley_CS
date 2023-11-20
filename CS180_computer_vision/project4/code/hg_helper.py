import cv2
import numpy as np
import os

def make_H_trans(inputs, outputs):
    
    '''
    P' = H * P
    
    |wx'|     | *, *, * |     |x|
    |wy'|  =  | *, *, * |  *  |y|
    |w  |     | *, *, * |     |1|
    
    using least squares to solve homegraphy
    
    |wx'|     | a, b, c |     |x|
    |wy'|  =  | d, e, f |  *  |y|
    |w  |     | g, h, 1 |     |1|
    
    x' = (a * x + b * y + c) / (g * x + h * y + 1)
    y' = (d * x + e * y + f) / (g * x + h * y + 1)
    
    (i=1 as scale=1) , 8 unknown parameters in total
    
    Set up a system of linear equations:
    
    Ah = b
    
    For example:
    given data pairs:
    (p1, p1'), (p2, p2'), (p3, p3'), ..., (pn, pn')
    
    try to find p * x1 + x2 = p'
    
    |p1, 1|              |p1'|
    |p2, 1|              |p2'|
    |p3, 1|  *  |x1|  =  |p3'|
    | ... |     |x2|     |p4'|
    |ph, 1|              |p5'|
    
    in homegraphy (Ah = b)
    
    |x, y, 1, 0 , 0 , 0, -x*x', -y*x'|           
    |0 , 0 , 0, x, y, 1, -x*y', -y*y'|  * h =  |x', y'|
    
    a * x + b * y + c - g*x*x' - h*y*x' = x'
    d * x + e * y + f - g*x*y' - h*y*y' = y'
    
    x' = (a * x + b * y + c) / (g * x + h * y + 1)
    y' = (d * x + e * y + f) / (g * x + h * y + 1)
    
    the format is the same
        
    '''
    
    # now we build A * h = b according to formula
    
    assert(len(inputs) == len(outputs))
    num_of_pts = len(inputs)
    
    A = np.zeros((num_of_pts * 2, 8))
    
    for i in range(0, num_of_pts * 2, 2):
        index = int(i // 2)
        x_hat, y_hat = outputs[index][0], outputs[index][1]
        x, y = inputs[index][0], inputs[index][1]
        A[i]   =  [x, y, 1, 0, 0, 0, -x*x_hat, -y*x_hat]
        A[i+1] =  [0, 0, 0, x, y, 1, -x*y_hat, -y*y_hat]
        
    b = outputs.ravel()
    
    #print(A)
    #print(b)
    
    return A, b    
    
    
def solve_homograpy(input_list, output_list):
    inputs = np.array(input_list)
    outputs = np.array(output_list)
    
    # A * h = b
    A, b =  make_H_trans(inputs, outputs)
    
    #print(A)
    #print(b)
    
    # now we calculate the H (a,b,c,d,e,f,g)
    # h = A-1 * b
    A_reverse = np.linalg.pinv(A)
    h = np.matmul(A_reverse, b)
    
    # the right format of h should be matrix:
    # | a, b, c|
    # | d, e, f|
    # | g, h, 1|
    
    h_vector = np.hstack((h, 1))
    h_matrix = np.reshape(h_vector, (3,3))
    
    return h_matrix

def add_1_for_homograph(points):
    
    '''
    input:
    
    [[x1, y1],
     [x2, y2],
       ...
     [xn, yn]]
     
    output:
    
    [[x1, y1, 1],
     [x2, y2, 1],
       ...
     [xn, yn, 1]]
       
       '''
    
    
    row, _ = points.shape
    return np.concatenate((points, np.ones((row, 1))), axis = 1)