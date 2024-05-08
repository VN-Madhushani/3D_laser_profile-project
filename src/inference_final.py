from __future__ import print_function
import argparse
import numpy as np
from util.util import to_categorical, compute_overall_iou, IOStream
import random
import numpy as np
from numpy import loadtxt
import torch
#import Functions as fn
import os
import glob
import front_min_point as fmp
from data_plotter_1_wspaths import Former, process_former_data,get_latest_files_info,get_file_info,copy_to_new_folder
import time
import online_data_gathering
from datetime import datetime
#from get_round_number import get_round
import matplotlib.pyplot as plt
from numpy import loadtxt
import sys
import open3d as o3d
from model.GDANet_ptseg import GDANet
from torch.autograd import Variable

def downsample(point_arr):
    desired_num_points = 2880
    #kdtree = cKDTree(point_arr)


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

def normal_generator(point_cloud):
    point_cloud_new = o3d.geometry.PointCloud()
    point_cloud_new.points = o3d.utility.Vector3dVector(point_cloud)

    #point_cloud_new.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=150))
    point_cloud_new.estimate_normals()
    normal = point_cloud_new.normals
    return normal

def test(args, io,points,norm_plt):

    points = np.reshape(points, (1, 2880, 3))
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



    #print(num_classes,'num_CLASSES')

    with torch.no_grad():
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50.

    return seg_pred,points

def seperate_fingers(points, np_array, i):
    finger_1 = []
    finger_2 = []
    finger_3 = []
    finger_4 = []
    finger_5 = []

    #for i in range(points.shape[0]):
    #print(np_arr.shape)
    #index_arr = np_array[i,:,:]  # Extract a single point cloud
    index_arr = np.argmax(np_array, axis = 1)


    x=0
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
        x+=1
        
    
    return finger_1, finger_2, finger_3, finger_4, finger_5

def visualize_segmented_results(np_array, points):
    for i in range(1):
        #index_arr = np_array[i,:,:]  # Extract a single point cloud
        index_arr = np.argmax(np_array, axis = 1)

        color_arr = []
        blue = []
        green = []
        red = []
        #print(points.shape)
        #print(points.shape)
        x = points[0,0,:]  # X-coordinates of points
        y = points[0,1,:]  # Y-coordinates of points
        z = points[0,2,:]  # Z-coordinates of points

        print('goes once')
        # x = (x-np.mean(x))
        # y = (y-np.mean(y))
        # z = (z-np.mean(z))
        # x = x*10/np.max(x)
        # y = y*10/np.max(y)
        # z = z*10/np.max(z)

        point_cloud_data_new = np.column_stack((x,y,z))
        x=0
        # for x in range (0, len(index_arr)):
        #print(index_arr)
        while x<len(index_arr):
            if index_arr[x]==0:
                red.append(0)
                blue.append(0)
                green.append(1)
                #print("Y0")
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

       
     

        #point_cloud = o3d.geometry.PointCloud()
        #point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data_new)
        #point_cloud.colors = o3d.utility.Vector3dVector(color_arr)
        #print(point_cloud_data_new)
        
        #o3d.visualization.draw_geometries([point_cloud]) 

        # return point_cloud
        return color_arr

def visualize_point_lines(color_arr, point_cloud, line_points, line):


    color_arr=np.array(color_arr)
    point_cloud=np.array(point_cloud)
    point_cloud=np.transpose(point_cloud)
    pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(color_arr)
    
    #print(color_arr)
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

if __name__ == "__main__":
   
    date = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    print("data grabbing.....")

    #prepared_data = online_data_gathering.gathering_data(date,current_round)
    # prepared_data = online_data_gathering.gathering_data()
    #prepared_data=np.loadtxt("C:\\Program Files\\Ansell\\Application\\Laser_controller_new_combined\\prepared_data\\2023_08_28__12_16_13_round_1.txt")
    prepared_data=np.loadtxt( "C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\dataset\\11110000\\former3_125_1R_1.txt")
    print("============")
    print("============")

    time.sleep(0.0001)
    start_0=time.time()
    
    

    former = Former()
    formers = process_former_data(prepared_data, former) 

    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='GDANet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--manual_seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=True,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()

    io = IOStream('checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    io.cprint(str(args))  

    
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
         io.cprint('Using GPU')
         if args.manual_seed is not None:
             torch.cuda.manual_seed(args.manual_seed)
             torch.cuda.manual_seed_all(args.manual_seed)



    i=1
 
    for x in range(len(formers)):

        start = time.time()
        point_data = np.array(formers[x])      
        #point_data = fn.z_outlier(point_data)
        #point_data = fn.filter(point_data)
        #point_data = fn.normalize_coordinates(point_data)
       # point_data = fn.filter(point_data)
        #print(point_data.shape)
        point_data = downsample(point_data)
        normal_data = np.array([normal_generator(point_data)])

      

        #print(normal_data.shape)
    
        seg_pred,point_data = test(args, io,point_data,normal_data)
    
        seg_pred_tensor = seg_pred[0]
        seg_pred_cpu = seg_pred_tensor.cpu()

     
        # seg_pred_cpu = seg_pred.cpu()
        np_arr = seg_pred_cpu.numpy()


        points_cpu = point_data.cpu()
        point_data = points_cpu.numpy() 

        #print(point_data.shape)


        #for i in range(0, len(np_arr)):
        for i in range(1):
            f1,f2,f3,f4,f5 =seperate_fingers(point_data,np_arr,i)
            f1 = np.array(f1)
            f2 = np.array(f2)
            f3 = np.array(f3)
            f4 = np.array(f4)
            f5 = np.array(f5)

            
            # key_point1, p1_max, p1_min = fn.fit_line_ransac(f1, n_iterations=100, threshold=0.1)
            # key_point2, p2_max, p2_min = fn.fit_line_ransac(f2, n_iterations=100, threshold=0.1)
            # key_point3, p3_max, p3_min = fn.fit_line_ransac(f3, n_iterations=100, threshold=0.1)
            # key_point4, p4_max, p4_min = fn.fit_line_ransac(f4, n_iterations=100, threshold=0.1)
            # key_point5, p5_max, p5_min = fn.fit_line_ransac(f5, n_iterations=100, threshold=0.1)


            color_arr = visualize_segmented_results(np_arr, point_data)



            #key_points = np.array((key_point1, key_point2,key_point3,key_point4,key_point5))

            point_data = np.squeeze(point_data)

            end = time.time()
            key_points=None
            #fn.visualize_point_lines(key_points,normal_data, color_arr, point_data)
            visualize_point_lines(color_arr,point_data,normal_data,key_points)
    
            print("time elapsed: ", end-start)


    end_0=time.time()
    # print("Total time: ",end_0-start_0)