from __future__ import print_function
import mmap
import time
import numpy as np
from numpy import loadtxt
import sys
import open3d as o3d
# from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import torch
from model.GDANet_ptseg import GDANet
from util.util import to_categorical, compute_overall_iou, IOStream
from tqdm import tqdm
from torch.autograd import Variable
import random
import os
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from scipy.optimize import minimize


###<<<<<<<<<removing noise>>>>>############################
# Using open 3D
def remove_noise(all_data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_data)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.1) #for tempory
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1) # for tempory_1/5-0.1
    all_data = np.asarray(cl.points)
    return all_data

# Using DBSCAN
def filter(cloud, eps = 150, min_samples = 15):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(cloud)
    filtered_point_cloud = cloud[clusters != -1]
    return filtered_point_cloud

# Using z score
def z_outlier(cloud, threshold = 3):
    mean = np.mean(cloud[:, 2])
    std = np.std(cloud[:, 2])
    outliers = []
    for i in range(len(cloud)):
        z = (float(cloud[i, 2]) - mean) / std
        if np.abs(z) < threshold:
            outliers.append(i)
    return np.array(cloud[outliers])


###<<<<<<<<<normalizing former data>>>>>############################
def normalize_coordinates(data_out):
    index=1
    data_out = data_out[data_out[:,index].argsort()]

    x_data = (data_out[:, 0] - np.mean(data_out[:, 0]))
    y_data = (data_out[:, 1] - np.mean(data_out[:, 1]))
    z_data = (data_out[:, 2] - np.mean(data_out[:, 2]))

    x_data = (x_data)*10/np.max(x_data) # seperate the x data
    y_data = (y_data)*10/np.max(y_data) # seperate the y data
    z_data = (z_data)*10 /np.max(z_data)# seperate the z data

    depth_map = np.sqrt(x_data**2 + y_data**2 + z_data**2)

    min_range = np.min(depth_map)
    max_range = np.max(depth_map)
    #print("min pixel : ",min_range)
    #print("max pixel : ",max_range)


    data_out = np.column_stack((x_data, y_data, z_data, ))
    return data_out
    
# def normal_generator(point_index, point_cloud):
#     point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=150))
#     points = np.asarray(point_cloud.points)  
#     normal = point_cloud.normals[point_index]
#     return normal

########<<<<<<<<<normal vector>>>>>############################
def normal_generator(point_cloud):
    # point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=150))
    # points = np.asarray(point_cloud.points)  
    # normal = point_cloud.normals[point_index]
    # print(np.array(point_cloud).shape)
    #print("Surface Normal at Point {}: {}".format(point_index, normal))
  



    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Estimate normals
    pcd.estimate_normals()
   
    # Get the normal vectors as a NumPy array
    normals = np.asarray(pcd.normals)
    #print(normals)
    #print(len(normals))

    return normals


########<<<<<<<<<downsample>>>>>############################oldd
# def downsample(point_arr):
#     desired_num_points = 4096
#     kdtree = cKDTree(point_arr)

#     sorted_indices = np.argsort(point_arr[:, 0])
#     sorted_point_cloud = point_arr[sorted_indices]

#     downsampled_point_cloud = []
#     initial_retention_prob = 1

#     for i in range(len(sorted_point_cloud)):
    
#         random_index = np.random.randint(len(sorted_point_cloud))
#         random_point = sorted_point_cloud[random_index]
#         x_value = random_point[0]
#         retention_prob = 1.1*initial_retention_prob * (x_value / sorted_point_cloud[-1, 0])

#         if np.random.rand() > retention_prob:
#             downsampled_point_cloud.append(random_point)

        
#         if len(downsampled_point_cloud) >= desired_num_points:
#             break

#     downsampled_point_cloud = np.array(downsampled_point_cloud)
#     downsampled_point_cloud = downsampled_point_cloud[np.argsort(downsampled_point_cloud[:,1])]

#     return downsampled_point_cloud
#######################
def downsample(point_arr):
    desired_num_points = 2880
    kdtree = cKDTree(point_arr)


    sorted_indices = np.argsort(point_arr[:, 0])
    sorted_point_cloud = point_arr[sorted_indices]


    downsampled_point_cloud = []


    initial_retention_prob = 1

    
    #for i in range(len(sorted_point_cloud)):
    while True:
        random_index = np.random.randint(len(sorted_point_cloud))
        random_point = sorted_point_cloud[random_index]

    
        x_value = abs(random_point[0])
        retention_prob = 1.1*initial_retention_prob * (x_value / sorted_point_cloud[-1, 0])

    

        thresh=np.random.rand()


        if thresh > retention_prob:
            downsampled_point_cloud.append(random_point)
        

        
        if len(downsampled_point_cloud) >= desired_num_points:
            break


    downsampled_point_cloud = np.array(downsampled_point_cloud)
    downsampled_point_cloud = downsampled_point_cloud[np.argsort(downsampled_point_cloud[:,1])]

    return downsampled_point_cloud


def test(args, io,points,norm_plt):
    
    num_part = 6
    device = torch.device("cuda" if args.cuda else "cpu")

    model = GDANet(num_part).to(device)
    print(device,'device')
    #io.cprint(str(model))

    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    model.eval()
    num_part = 6
    num_classes = 16
    label = torch.tensor([[[0]]])
    label = label.to(device)

  
    #for batch_id in tqdm(range(1), total=1, smoothing=0.9):   
    points,norm_plt= Variable(torch.from_numpy(points).float()), Variable(torch.from_numpy(norm_plt).float())

    points = points.transpose(2, 1)
    norm_plt = norm_plt.transpose(2, 1)
    points,  norm_plt = points.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

    #print(points,'points',points.shape)
   # print(norm_plt,'normal',norm_plt.shape)
    #print(label,'labels')
    #print(num_classes,'num_CLASSES')
    with torch.no_grad():
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50.

    
    return seg_pred,points



########<<<<<<<<<visualize the segmentation>>>>>############################
def visualize_segmented_results(np_array, points):
    #point_clouds_cpu = seg_pred

    #np_array = point_clouds_cpu.numpy()
    #points_cpu = points.cpu()
    #point_cloud_data = points_cpu.numpy() 
    #print("kkkkkkkkkkkkkkkkkkkkkkkkk: ",points.shape)
    for i in range(points.shape[0]):
        index_arr = np_array[i,:,:]  # Extract a single point cloud
        index_arr = np.argmax(index_arr, axis = 1)

        color_arr = []
        blue = []
        green = []
        red = []
        x = points[i,0,:]  # X-coordinates of points
        y = points[i,1,:]  # Y-coordinates of points
        z = points[i,2,:]  # Z-coordinates of points

    
        # x = (x-np.mean(x))
        # y = (y-np.mean(y))
        # z = (z-np.mean(z))
        # x = x*10/np.max(x)
        # y = y*10/np.max(y)
        # z = z*10/np.max(z)

        point_cloud_data_new = np.column_stack((x,y,z))
        x=0
        # for x in range (0, len(index_arr)):
        while x<len(index_arr):
            if index_arr[x]==0:
                red.append(0)
                blue.append(0)
                green.append(1)
                print("Y0")
                #print(point_cloud_data_new.shape)
                # point_cloud_data_new = np.delete(point_cloud_data_new,x,0)
                # index_arr = np.delete(index_arr,x,0)
                # continue
            elif index_arr[x]==1:
                red.append(0)
                blue.append(1)
                green.append(1)
            elif index_arr[x]==2:
                red.append(1)
                blue.append(0)
                green.append(1)
            elif index_arr[x]==3:
                red.append(1)
                blue.append(0)
                green.append(0)
            elif index_arr[x]==4:            
                red.append(1)
                blue.append(1)
                green.append(0)
            elif index_arr[x]==5:
                red.append(0)
                blue.append(1) 
                green.append(0)
            x+=1
    
       #VISUALIZE THE SEGMENTATION...................................
        color_arr = np.column_stack((red,green,blue))
        #print("color array = ", color_arr.shape)
        #print("point array = ", point_cloud_data_new.shape)

       
     

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data_new)
        point_cloud.colors = o3d.utility.Vector3dVector(color_arr)
        #print(point_cloud_data_new)
        
        #o3d.visualization.draw_geometries([point_cloud]) 

        # return point_cloud
        return color_arr




########<<<<<<<<<Checkpoint save>>>>>############################
def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)

    if not args.eval:  # backup the running files
        os.system('cp main_cls.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
        os.system('cp model/GDANet_ptseg.py checkpoints' + '/' + args.exp_name + '/' + 'GDANet_ptseg.py.backup')
        os.system('cp util.GDANet_util.py checkpoints' + '/' + args.exp_name + '/' + 'GDANet_util.py.backup')
        os.system('cp util.data_util.py checkpoints' + '/' + args.exp_name + '/' + 'data_util.py.backup')
      



########<<<<<<<<<seperate finger segments>>>>>############################
def seperate_fingers(points, np_array, i):
    

    
    finger_1 = []
    finger_2 = []
    finger_3 = []
    finger_4 = []
    finger_5 = []


    #for i in range(points.shape[0]):
    index_arr = np_array[i,:,:]  # Extract a single point cloud
    index_arr = np.argmax(index_arr, axis = 1)



    x=0
    # for x in range (0, len(index_arr)):
    #print(index_arr)
    #print(points.shape,'points')
    while x<len(index_arr):
        if index_arr[x]==1:
            #finger_1.append(points[i,x,:])
            finger_1.append(points[i,x,:])
        if index_arr[x]==2:
            #finger_2.append(points[i,x,:])
            finger_2.append(points[i,x,:])
        if index_arr[x]==3:
            #finger_3.append(points[i,x,:])
            finger_3.append(points[i,x,:])
        if index_arr[x]==4:            
            #finger_4.append(points[i,x,:])
            finger_4.append(points[i,x,:])
        if index_arr[x]==5:
            #finger_5.append(points[i,x,:])
            finger_5.append(points[i,x,:])
        #print(x)
        x+=1
        
    
    return finger_1, finger_2, finger_3, finger_4, finger_5

    





def fit_line_ransac(pointcloud, n_iterations=100, threshold=0.1):


    sorted_indices = np.argsort(pointcloud[:, 0])
    pointcloud = pointcloud[sorted_indices]
    
    l1 = len(pointcloud)-1
    l2 = len(pointcloud)-21
    xmin = np.mean(pointcloud[l2:l1 ,0])
    ymin = np.mean(pointcloud[l2:l1 ,1])
    zmin = np.mean(pointcloud[l2:l1 ,2])

    x_mean = np.mean(pointcloud[:,0])
    y_mean = np.mean(pointcloud[:,1])
    z_mean = np.mean(pointcloud[:,2])

    sorted_indices = np.argsort(pointcloud[:, 1])
    pointcloud = pointcloud[sorted_indices]

    best_line = None
    best_inliers_count = 0
    
    for i in range(n_iterations):
        # Step 1: Randomly select two points from the pointcloud
        if i==0:
            p1 = np.array([xmin, ymin, zmin])
            p2 = np.array([x_mean,y_mean,z_mean])
        
        else:
            indices = np.random.choice(len(pointcloud), 2)
            p1, p2 = pointcloud[indices]

        
        # Step 2: Fit a line through the two points
        line_direction = p2 - p1

        # Calculate the coefficients of the line
        a, b, c = line_direction
        d = -(a*p1[0] + b*p1[1] + c*p1[2])

        # Step 3: Compute the residual errors for all the points in the pointcloud
        numerator=np.abs(pointcloud.dot(np.array([a,b,c])) + d)
        denominator=np.sqrt(a**2 + b**2 + c**2)
        threshold<=1e-5
        if denominator>=threshold:
            distances = numerator /denominator
        else:
            distances=numerator/threshold

        # Step 4: Count the number of inliers and update the best-fitting line if necessary
        inliers_count = np.sum(distances < threshold)
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_line = a, b, c, d



            #print(a,b,c,d)


        
    point_max = [0,0,0]
    point_min = [0,0,0]
    eqn = 0


    sorted_indices = np.argsort(pointcloud[:, 0])
    pointcloud = pointcloud[sorted_indices]
    #print(pointcloud.shape)
    
    for i in range(len(pointcloud)-1, 0,-1 ):
        eqn = a*pointcloud[i,0] + b*pointcloud[i,1] + c*pointcloud[i,2] + d
        #print("min eqn: ",eqn)
        if eqn == 0:
            point_min[0] = pointcloud[i,0]
            point_min[1] = pointcloud[i,1]
            point_min[2] = pointcloud[i,2]
            break
        else:
            continue

    for i in range(0, len(pointcloud)):
        eqn = a*pointcloud[i,0] + b*pointcloud[i,1] + c*pointcloud[i,2] + d
        #print("max eqn:",eqn)
        if eqn == 0:
            point_max[0] = pointcloud[i,0]
            point_max[1] = pointcloud[i,1]
            point_max[2] = pointcloud[i,2]
            break
        else:
            continue

    
    
    point_max = [x_mean,y_mean,z_mean]
    point_min = [xmin,ymin,zmin]
    #print("$$$$$$$$$$$$$$$$$$$$$$$",point_max)
    #print("$$$$$$$$$$$$$$$$$$$$$$$",point_min)
    #return best_line, np.array(point_max),np.array(point_min)
    return best_line, point_max, point_min


def visualize_point_lines(color_arr, point_cloud, line_points, line):
    
    #print(color_arr)
    color_arr=np.array(color_arr)
    #point_cloud=point_cloud.tolist()
    pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.points = o3d.utility.Vector3dVector(np.array(point_cloud, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(color_arr)
    

    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(key_point)
    # pcd1.colors = o3d.utility.Vector3dVector(key_colors)

    line_points = np.array(line_points)
    #print(line_points,'line_points')
    line = np.array(line)
    #print(line,'line')
    #print("line is::::::::",line)
    # Create a line set from the fitted line
    #line_set = o3d.geometry.LineSet()
    #line_set.points = o3d.utility.Vector3dVector(line_points)  # Two points for the line
    #line_set.lines = o3d.utility.Vector2iVector(line)
    

    
    #print("line points: ",line_points)
    
    # Visualize the point cloud and the line set
    o3d.visualization.draw_geometries([pcd])

    

      


#################################################
###################################################
###################################################

