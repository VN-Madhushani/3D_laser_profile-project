from __future__ import print_function
import argparse
import torch
import time
import numpy as np
from util.util import to_categorical, compute_overall_iou, IOStream
import random
import numpy as np
from numpy import loadtxt
import torch
import Functions as fn
import os
import glob
# import key_point_functions as kp
import test as tt
import matplotlib.pyplot as plt
# import test2 as tt2
# import test3 as tt3

  

  
if __name__ == "__main__":
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
    checkpoints_path="C:\\Program Files\\Ansell\\Application\\src_process\\checkpoints"
    io = IOStream(checkpoints_path+"\\" + args.exp_name + '/%s_test.log' % (args.exp_name))
    #io.cprint(str(args))

    
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)


    

    #data_path ="D:\\induwara\\GDANet-main\\data\\former_data\\new"
    data_path="C:\\Program Files\\Ansell\\Application\\src_process\\data\\new_former_inference"
    
    
    # for data_name in os.listdir(data_path):
    #     file = os.path.join(data_path, data_name)


    for filename in glob.glob(data_path + '/*.txt'):
        data = np.loadtxt(filename)
        name=(filename.split('\\')[-1]).split('.')[0]
        point_data = np.array(data)
        # point_data =data[:,:, 0:3].astype('float')
        # normal_data = data[:,:,3:6].astype('float')
        
        
        start = time.time()
        point_data = fn.z_outlier(point_data)
        #point_data = fn.filter(point_data)
        point_data = fn.normalize_coordinates(point_data)
        point_data = fn.downsample(point_data)
        
        normal_data = np.array([fn.normal_generator(point_data)])

        #print("normals: ",normal_data)

        point_data = np.array([point_data])
        #print(normal_data.shape)

        seg_pred,points= fn.test(args, io,point_data,normal_data)
        #print(seg_pred)
        print(seg_pred.shape)
        end = time.time()

        seg_pred_cpu = seg_pred.cpu()
        np_arr = seg_pred_cpu.numpy()
        
        points_cpu = points.cpu()
        point_data = points_cpu.numpy() 
        
    
       # print(len(np_arr),'length')
        

        for i in range(0, len(np_arr)):
            f1,f2,f3,f4,f5 = fn.seperate_fingers(point_data,np_arr,i)
            f1 = np.array(f1)
            f2 = np.array(f2)
            f3 = np.array(f3)
            f4 = np.array(f4)
            f5 = np.array(f5)
            #print(f4.shape)

            #print(f1)

            file_path = "example.txt"

       

            line_1, p1_max, p1_min = fn.fit_line_ransac(f1, n_iterations=100, threshold=0.1)
            line_2, p2_max, p2_min = fn.fit_line_ransac(f2, n_iterations=100, threshold=0.1)
            line_3, p3_max, p3_min = fn.fit_line_ransac(f3, n_iterations=100, threshold=0.1)
            line_4, p4_max, p4_min = fn.fit_line_ransac(f4, n_iterations=100, threshold=0.1)
            line_5, p5_max, p5_min = fn.fit_line_ransac(f5, n_iterations=100, threshold=0.1)
            
          
            line_points = np.row_stack((p1_max, p1_min, p2_max, p2_min, p3_max, p3_min, p4_max, p4_min, p5_max, p5_min))
            line_set = np.array(([0,1],[2,3],[4,5],[6,7],[8,9]))
            # line_points = np.array(line_points)
            # line_points = np.transpose(line_points)
            #print("line points::::::::::::", line_points.shape)
            #print(point_data)
            #print(len(point_data))
            color_arr,point_data = fn.visualize_segmented_results(np_arr, point_data)

            
            # p1 = kp.contours(f1)
            # p2 = kp.contours(f2)
            # p3 = kp.contours(f3)
            # p4 = kp.contours(f4)
            # p5 = kp.contours(f5)

            

            # p1 = tt.contours(f1)
            # p2 = tt.contours(f2)
            # p3 = tt.contours(f3)
            # p4 = tt.contours(f4)
            # p5 = tt.contours(f5)


            # p1 = tt3.front(f1)
            # p2 = tt3.front(f2)
            # p3 = tt3.front(f3)
            # p4 = tt3.front(f4)
            # p5 = tt3.front(f5)




            # key_points = np.row_stack((p1,p2,p3,p4,p5))
            # key_colors = [[0,1,1],[0,0,0],[0,0,0],[0,0,0],[0,0,1]]

            


           

            file_path_colors = "color\\"+name+".txt"

            #np.savetxt(file_path_colors, color_arr, fmt='%1.6f', delimiter=' ')

            file_path_points = "point\\"+name+".txt"

            #snp.savetxt(file_path_points, point_data, fmt='%1.6f', delimiter=' ')
           

            #point_data = np.squeeze(point_data)
            fn.visualize_point_lines(color_arr, point_data, line_points, line_set)
            
            ###fn.visualize_point_and_line( point_data, line_1)
        
            print("time elapsed: ", end-start)

            

