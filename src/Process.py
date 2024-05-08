from __future__ import print_function
import argparse
import numpy as np
from util.util import to_categorical, compute_overall_iou, IOStream
import random
import numpy as np
from numpy import loadtxt
import torch
import Functions as fn
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
from datetime import date
import json



def main():


    date = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    #input raw data of sensor
    file_path = 'C:/Program Files/Ansell/Application/src_process/data/2023_12_01__16_42_13_round_1.txt'
    data = np.loadtxt(file_path)


    #Get rack size as an user input
    rack_size = input("Enter racksize: ")
    # rack_size = 5


    json_file_path = 'C:/Program Files/Ansell/Application/src_process/src/former_y_ranges.json' 
    all_y_ranges = fn.read_json(json_file_path)

    if f"FK10{rack_size}" in all_y_ranges:
        former_y_ranges = all_y_ranges[f"FK10{rack_size}"]
    else:
        print(f"Y-coordinate ranges for rack {rack_size} not found in the JSON file.")
        return


    # ********Seperate into formers***************

    fn.visualize_rack(data, former_y_ranges, rack_size)
    segmented_data = fn.save_segmented_data(data, former_y_ranges)
    formers=fn.save_former_data(date,segmented_data)


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

    io = IOStream('C:/Program Files/Ansell/Application/src_process/src/checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    # io = IOStream("C:/Program Files/Ansell/Application/src_process/src/checkpoints/GDANet/GDANet_test.log")
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
            

        seg_pred_cpu = seg_pred.cpu()
        np_arr = seg_pred_cpu.numpy()



            
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

            fn.visualize_point_lines(key_points,normal_data, color_arr, point_data)

            
            print("time elapsed: ", end-start)


    end_0=time.time()
    # print("Total time: ",end_0-start_0)

if __name__ == "__main__":
    segmented_data_result = main()
    



