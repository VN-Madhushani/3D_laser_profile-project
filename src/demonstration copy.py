

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import argparse
import time
from util.util import to_categorical, compute_overall_iou, IOStream
import random
import torch
import Functions as fn
import os
import glob
import front_min_point as fmp
from numpy import loadtxt
import sys
import open3d as o3d
from datetime import date
import online_data_gathering_copy
from matplotlib.colors import ListedColormap

def segment_points(points, object_y_ranges):
    y = points[:, 1]
    section_ids = np.zeros(len(points), dtype=int)

    for obj_id, y_range in enumerate(object_y_ranges):
        lower_bound, upper_bound = y_range
        in_object = np.logical_and(y >= lower_bound, y < upper_bound)
        section_ids[in_object] = obj_id + 1

    return section_ids

def save_segmented_data(points, object_y_ranges):
    section_ids = segment_points(points, object_y_ranges)
    segmented_data = [None] * len(object_y_ranges)

    for obj_id, y_range in enumerate(object_y_ranges, start=1):
        obj_indices = np.where(section_ids == obj_id)
        obj_data = points[obj_indices]
        segmented_data[obj_id - 1] = obj_data
    return tuple(segmented_data)

def visualize_rack(points, object_y_ranges, rack_size):
    section_ids = segment_points(points, object_y_ranges)

    fig = plt.figure(figsize=(15, 12))

    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xticks(np.arange(0, 2240, 50))
    ax.set_yticks(np.arange(0, 700, 50))
    ax.grid()


    colors = ['blue', 'green', 'orange', 'purple', 'red','yellow','pink','gray']
    custom_cmap = ListedColormap(colors)
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=section_ids, cmap='plasma')
    ax.scatter(points[:, 0], points[:, 1], c=section_ids, cmap=custom_cmap)
    ax.set_title(f"Segmented Rack Left FK10{rack_size}(Size: {rack_size})")

    for arm in object_y_ranges:
        end_point=arm[1]
        y=[end_point]*len(points[:,0])
        plt.plot(points[:,0],y,color='red')


    plt.show()

def save_arrays_to_files(segmented_data, file_prefix='segmented_data'):
    for i, data_array in enumerate(segmented_data):
        file_path = f"{file_prefix}_{i + 1}.txt"
        np.savetxt(file_path, data_array)
        print(f"Saved {file_path}")

def save_former_data(date,segmented_data,count,rack):
    print(count)
    former_1 = []
    former_2 = []
    former_3 = []
    former_4 = []
    former_5 = []
    former_6 = []
    former_7 = []
    former_8 = []

    for i, data_array in enumerate(segmented_data):
        if i == 0:
            former_1 = data_array
        elif i == 1:
            former_2 = data_array
        elif i == 2:
            former_3 = data_array
        elif i == 3:
            former_4 = data_array
        elif i == 4:
            former_5 = data_array
        elif i == 5:
            former_6 = data_array
        elif i == 6:
            former_7 = data_array
        elif i == 7:
            former_8 = data_array



    # round= get_round(date)
        
    np.savetxt("D:/Laser_controller_new_combined/new_Data/prepared_data_demo/former_data_demo/former_1_round_{}_{}_{}.txt".format(date,count,rack),former_1)
    np.savetxt("D:/Laser_controller_new_combined/new_Data/prepared_data_demo/former_data_demo/former_2_round_{}_{}_{}.txt".format(date,count,rack),former_2)
    np.savetxt("D:/Laser_controller_new_combined/new_Data/prepared_data_demo/former_data_demo/former_3_round_{}_{}_{}.txt".format(date,count,rack),former_3)
    np.savetxt("D:/Laser_controller_new_combined/new_Data/prepared_data_demo/former_data_demo/former_4_round_{}_{}_{}.txt".format(date,count,rack),former_4)
    np.savetxt("D:/Laser_controller_new_combined/new_Data/prepared_data_demo/former_data_demo/former_5_round_{}_{}_{}.txt".format(date,count,rack),former_5)
    np.savetxt("D:/Laser_controller_new_combined/new_Data/prepared_data_demo/former_data_demo/former_6_round_{}_{}_{}.txt".format(date,count,rack),former_6)
    np.savetxt("D:/Laser_controller_new_combined/new_Data/prepared_data_demo/former_data_demo/former_7_round_{}_{}_{}.txt".format(date,count,rack),former_7)
    np.savetxt("D:/Laser_controller_new_combined/new_Data/prepared_data_demo/former_data_demo/former_8_round_{}_{}_{}.txt".format(date,count,rack),former_8)


    return former_1, former_2, former_3, former_4, former_5, former_6, former_7, former_8

def get_formers(points,racksize):
    hand_ranges=[]
    y = points[:, 1]
    start=y[0]
    last_val=y[-1]
    #print(start,last_val)

    if racksize==0:arm,tol=135,67.5
    elif racksize==1:arm,tol=140,55
    elif racksize==2:arm,tol=140,62.5
    elif racksize==3:arm,tol=147.5,52.5
    elif racksize==4:arm,tol=162.5,40
    elif racksize==5:arm,tol=172.5,28.75

    gap=arm+tol
    for former in range(8):
            if former==0: 
                hand_ranges.append([start,start+gap-tol/2])
                start+=gap-tol/2
            else:
                hand_ranges.append([start,start+gap])
                start+=gap
    #print(hand_ranges)
    return hand_ranges



def main():

        date = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        print("data grabbing.....")
        
        print("============")
        current_round=1
        #prepared_data = online_data_gathering_copy.gathering_data(date,current_round)






        #formerdata_folder_path="D:\\Laser_controller_new_combined\\new_Data\\prepared_data_demo"
        
        file_count=0
        # for file_name in os.listdir(formerdata_folder_path):
        #filepath = os.path.join(formerdata_folder_path, file_name)
       # data = prepared_data
    

    #input raw data of sensor
        file_path = 'D:\\Laser_controller_new_combined\\new_Data\\prepared_data_demo\\2024_01_08__14_28_09_round_1_1R.txt'
        data = np.loadtxt(file_path)
        #data = prepared_data

    #Get rack size as an user input
        rack_size = 1
        rack_type='1R'
        print(rack_type)

        fig = plt.figure()
        data_out=data
        ax= fig.add_subplot(111,projection='3d')
        ax.scatter(data_out[:, 0], data_out[:, 1], data_out[:, 2], s=1)
        plt.show()
        # ax.scatter(cloud1[:, 0], cloud1[:, 1], c='blue', marker='o', label='Cloud 1')




        fig = plt.figure()
        cloud1=data
       
        former_y_ranges=get_formers(data,rack_size)
        line_set=[]
        for former in former_y_ranges:
            y=former[1]*np.ones_like(cloud1[:,0])
            plt.plot(cloud1[:,0], y,linewidth=2.5)
            line_set.append(y)

        #plt.plot(cloud1[:,0], line_set[1])
        ax= fig.add_subplot(111)
        #section_ids = segment_points(data, object_y_ranges)
        colors = ['blue', 'green', 'orange', 'purple', 'red','yellow','pink','gray']
        custom_cmap = ListedColormap(colors)
        # ax.scatter(cloud1[:, 0], cloud1[:, 1], c='blue', marker='o', label='Cloud 1')
        ax.scatter(cloud1[:, 0], cloud1[:, 1],marker='o', label='Cloud 1',cmap='custom_cmap')
        y=cloud1[:,1]

        last_val=y[-1]
        print(last_val-y[0],'range')
        print(last_val,'Final data:')
        print(y[0],'first data')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        #ax.set_xticks(np.arange(0, 2240, 50))
        #ax.set_yticks(np.arange(0, 1850, 5))



        ax.grid(True)

        ax.legend()
        plt.show()

        """
        json_file_path = 'D:/Laser_controller_new_combined/former_y_ranges.json' 
        all_y_ranges = read_json(json_file_path)

        if f"FK10{rack_size}" in all_y_ranges:
            former_y_ranges = all_y_ranges[f"FK10{rack_size}"]
        else:
            print(f"Y-coordinate ranges for rack {rack_size} not found in the JSON file.")
            return
        """

        
        # ********Seperate into formers***************

        #visualize_rack(data, former_y_ranges, rack_size)
       
        segmented_data = save_segmented_data(data, former_y_ranges)
        print(former_y_ranges)
        formers=save_former_data(date,segmented_data,file_count,rack_type)
        file_count+=1


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

        # _init_(args)

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
            point_data = fn.z_outlier(point_data)
            #point_data = fn.filter(point_data)
            point_data = fn.normalize_coordinates(point_data)
        # point_data = fn.filter(point_data)
            point_data = fn.downsample(point_data)
            
            normal_data = np.array([fn.normal_generator(point_data)])

            point_data = np.array([point_data])
            #print(normal_data.shape)

            seg_pred = fn.test(args, io,point_data,normal_data)
            seg_pred_tensor = seg_pred[0]
            seg_pred_cpu = seg_pred_tensor.cpu()

            # seg_pred_cpu = seg_pred.cpu()
            np_arr = seg_pred_cpu.numpy()

            seg_pred_tensor = seg_pred[0]
            seg_pred_cpu = seg_pred_tensor.cpu()



            full_colors=[]
            full_points=[]
            
            for i in range(0, len(np_arr)):
                f1,f2,f3,f4,f5 = fn.seperate_fingers(point_data,np_arr,i)
                f1 = np.array(f1)
                f2 = np.array(f2)
                f3 = np.array(f3)
                f4 = np.array(f4)
                f5 = np.array(f5)



                
                key_point1, p1_max, p1_min = fn.fit_line_ransac(f1, n_iterations=100, threshold=0.1)
                key_point2, p2_max, p2_min = fn.fit_line_ransac(f2, n_iterations=100, threshold=0.1)
                key_point3, p3_max, p3_min = fn.fit_line_ransac(f3, n_iterations=100, threshold=0.1)
                key_point4, p4_max, p4_min = fn.fit_line_ransac(f4, n_iterations=100, threshold=0.1)
                key_point5, p5_max, p5_min = fn.fit_line_ransac(f5, n_iterations=100, threshold=0.1)


                color_arr = fn.visualize_segmented_results(np_arr, point_data)



                key_points = np.array((key_point1, key_point2,key_point3,key_point4,key_point5))

                point_data = np.squeeze(point_data)

                end = time.time()

                #fn.visualize_point_lines(key_points,normal_data, color_arr, point_data)
                full_colors.append(color_arr)
                full_points.append(point_data)
                fn.visualize_point_lines(color_arr,point_data)
        
                print("time elapsed: ", end-start)
            
            

        end_0=time.time()
        # print("Total time: ",end_0-start_0)

if __name__=="__main__":
    main()