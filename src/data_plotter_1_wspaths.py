import ctypes
import mmap
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
import sys
import open3d as o3d
import os
#import online_data_gathering
from online_data_gathering import gathering_data
import glob
from operator import itemgetter
import shutil
from datetime import datetime


np.set_printoptions(threshold=sys.maxsize)
########################################################

#read text file into NumPy array
#folder_path = "D:\\UoP\Midushan\\Laser_profiler_operation_finalized\\Laser_controller\\data\\rack_data" --> newset one
#folder_path = "D:\\Nisansala\\inference data_31_10\\test_11_10\\prepared_data"



class Former:
    def __init__(self):
        self.cases = {}

    def case(self, case_name):
        def case_decorator(func):
            self.cases[case_name] = func
            return func
        return case_decorator

    def default(self, *args, **kwargs):
        raise ValueError(f"Unhandled case: {args}")

    def __call__(self, case_name, *args, **kwargs):
        case_function = self.cases.get(case_name, self.default)
        return case_function(*args, **kwargs)

# former = Former()        
# @former.case('case1') 
# def case_1():
#     former_1.append(data)

# @former.case('case2') 
# def case_2():
#     former_2.append(data)

# @former.case('case3') 
# def case_3():
#     former_3.append(data)

# @former.case('case4')
# def case_4():
#     former_4.append(data)

# @former.case('case5')
# def case_5():
#     former_5.append(data)

# @former.case('case6')
# def case_6():
#     former_6.append(data)

# @former.case('case7')
# def case_7():
#     former_7.append(data)

# @former.case('case8')
# def case_8():
#     former_8.append(data)

def get_file_info(folder_path):
    files = os.listdir(folder_path)

    if files:
        file = max(files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
        filepath = os.path.join(folder_path, file)
        return file, filepath
    else:
        return None, None


def copy_to_new_folder(source_file, destination_folder):
    # Create a new folder with current date and time
    current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    new_folder_name = f"former_data_{current_datetime}"
    new_folder_path = os.path.join(destination_folder, new_folder_name)

    # Make sure the destination folder exists, create if not
    os.makedirs(new_folder_path, exist_ok=True)

    # Get the filename from the source file path
    file_name = os.path.basename(source_file)

    # Create the destination file path
    destination_file_path = os.path.join(new_folder_path, file_name)

    # Copy the file to the new destination
    shutil.copy(source_file, destination_file_path)

    return new_folder_path


def get_latest_files_info(folder_path, num_files):
    files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    if files:
        file_info_list = [(f, os.path.getmtime(f)) for f in files]
        file_info_list.sort(key=itemgetter(1), reverse=True)

        latest_files = file_info_list[:num_files]
        latest_filepaths = [f for f, _ in latest_files]

        return latest_files,latest_filepaths
    else:
        return []



# def process_former_data(file_path, former):
#     former=Former()
       
#     @former.case('case1') 
#     def case_1():
#         former_1.append(data)

#     @former.case('case2') 
#     def case_2():
#         former_2.append(data)

#     @former.case('case3') 
#     def case_3():
#         former_3.append(data)

#     @former.case('case4')
#     def case_4():
#         former_4.append(data)

#     @former.case('case5')
#     def case_5():
#         former_5.append(data)

#     @former.case('case6')
#     def case_6():
#         former_6.append(data)

#     @former.case('case7')
#     def case_7():
#         former_7.append(data)

#     @former.case('case8')
#     def case_8():
#         former_8.append(data)

        
#     file_number = 1
#     file_path = os.path.dirname(file_path)
#     for file_name in os.listdir(file_path):
#         former_1 = []
#         former_2 = []
#         former_3 = []
#         former_4 = []
#         former_5 = []
#         former_6 = []
#         former_7 = []
#         former_8 = []
    
    
#         z = 1
#         file = os.path.join(file_path, file_name)
#         data_out = np.loadtxt(file)
#         index=1
#         data_out = data_out[data_out[:,index].argsort()]
#         yp = data_out[0,1]


#         for x in range(0, len(data_out)):
#             dif = data_out[x,1] - yp
#             if dif>25:
#                 z = z+1
                
#             else:
#                 z=z
#             data = data_out[x,:]
#             case = "case"+ str(z)
#             #print(case)
#             former_upload =  former(case)
#         # print(yp)
#             yp = data_out[x,1]

#         former_1 = np.array(former_1)
#         former_2 = np.array(former_2)
#         former_3 = np.array(former_3)
#         former_4 = np.array(former_4)
#         former_5 = np.array(former_5)
#         former_6 = np.array(former_6)
#         former_7 = np.array(former_7)
#         former_8 = np.array(former_8)

    
# #midushan's code
#     #np.savetxt("D:/UoP/Midushan/Laser_profiler_operation_finalized/Laser_controller/data/former_data/f{}_round_1.txt".format(file_number),former_1)
#     #np.savetxt("D:/UoP/Midushan/Laser_profiler_operation_finalized/Laser_controller/data/former_data/f{}_round_2.txt".format(file_number),former_2)
#     #np.savetxt("D:/UoP/Midushan/Laser_profiler_operation_finalized/Laser_controller/data/former_data/f{}_round_3.txt".format(file_number),former_3)
#     #np.savetxt("D:/UoP/Midushan/Laser_profiler_operation_finalized/Laser_controller/data/former_data/f{}_round_4.txt".format(file_number),former_4)
#     #np.savetxt("D:/UoP/Midushan/Laser_profiler_operation_finalized/Laser_controller/data/former_data/f{}_round_5.txt".format(file_number),former_5)
#     #np.savetxt("D:/UoP/Midushan/Laser_profiler_operation_finalized/Laser_controller/data/former_data/f{}_round_6.txt".format(file_number),former_6)
#     #np.savetxt("D:/UoP/Midushan/Laser_profiler_operation_finalized/Laser_controller/data/former_data/f{}_round_7.txt".format(file_number),former_7)
#     #np.savetxt("D:/UoP/Midushan/Laser_profiler_operation_finalized/Laser_controller/data/former_data/f{}_round_8.txt".format(file_number),former_8)
    
# #updated path
#     np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f{}_round_1.txt".format(file_number),former_1)
#     np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f{}_round_2.txt".format(file_number),former_2)
#     np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f{}_round_3.txt".format(file_number),former_3)
#     np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f{}_round_4.txt".format(file_number),former_4)
#     np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f{}_round_5.txt".format(file_number),former_5)
#     np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f{}_round_6.txt".format(file_number),former_6)
#     np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f{}_round_7.txt".format(file_number),former_7)
#     np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f{}_round_8.txt".format(file_number),former_8)

#     file_number+=1

# #commented on 11_22 to optimize the code 
#     # fig1 = plt.figure(figsize=(15, 12))
#     # ax2= fig1.add_subplot(111, projection='3d')
#     # ax2.scatter(former_1[:, 0], former_1[:, 1], former_1[:, 2], s=1)

#     # fig2 = plt.figure(figsize=(15, 12))
#     # ax2= fig2.add_subplot(111, projection='3d')
#     # ax2.scatter(former_2[:, 0], former_2[:, 1], former_2[:, 2], s=1)

#     # fig3 = plt.figure(figsize=(15, 12))
#     # ax3= fig3.add_subplot(111, projection='3d')
#     # ax3.scatter(former_3[:, 0], former_3[:, 1], former_3[:, 2], s=1)

#     # fig4 = plt.figure(figsize=(15, 12))
#     # ax4= fig4.add_subplot(111, projection='3d')
#     # ax4.scatter(former_4[:, 0], former_4[:, 1], former_4[:, 2], s=1)

#     # fig5 = plt.figure(figsize=(15, 12))
#     # ax5= fig5.add_subplot(111, projection='3d')
#     # ax5.scatter(former_5[:, 0], former_5[:, 1], former_5[:, 2], s=1)

#     # fig6 = plt.figure(figsize=(15, 12))
#     # ax6= fig6.add_subplot(111, projection='3d')
#     # ax6.scatter(former_6[:, 0], former_6[:, 1], former_6[:, 2], s=1)

#     # fig7 = plt.figure(figsize=(15, 12))
#     # ax7= fig7.add_subplot(111, projection='3d')
#     # ax7.scatter(former_7[:, 0], former_7[:, 1], former_7[:, 2], s=1)

#     # fig8 = plt.figure(figsize=(15, 12))
#     # ax8= fig8.add_subplot(111, projection='3d')
#     # ax8.scatter(former_8[:, 0], former_8[:, 1], former_8[:, 2], s=1)
#     # plt.show()
            


#     return former_1, former_2, former_3, former_4, former_5, former_6, former_7, former_8

def process_former_data(data_out, former):

    #former=Former()
       
    @former.case('case1') 
    def case_1():
        former_1.append(data)

    @former.case('case2') 
    def case_2():
        former_2.append(data)

    @former.case('case3') 
    def case_3():
        former_3.append(data)

    @former.case('case4')
    def case_4():
        former_4.append(data)

    @former.case('case5')
    def case_5():
        former_5.append(data)

    @former.case('case6')
    def case_6():
        former_6.append(data)

    @former.case('case7')
    def case_7():
        former_7.append(data)

    @former.case('case8')
    def case_8():
        former_8.append(data)


    

    former_1 = []
    former_2 = []
    former_3 = []
    former_4 = []
    former_5 = []
    former_6 = []
    former_7 = []
    former_8 = []

    z = 1
    data_out = data_out[data_out[:, 1].argsort()]
    yp = data_out[0, 1]
    file_number=1
    for x in range(0, len(data_out)):
        dif = data_out[x, 1] - yp
        if dif > 20:
            z = z + 1
        else:
            z = z
        data = data_out[x, :]
        case = "case" + str(z)
        former_upload = former(case)
        
        yp = data_out[x, 1]

    former_1 = np.array(former_1)
    former_2 = np.array(former_2)
    former_3 = np.array(former_3)
    former_4 = np.array(former_4)
    former_5 = np.array(former_5)
    former_6 = np.array(former_6)
    former_7 = np.array(former_7)
    former_8 = np.array(former_8)

    # np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f1_round_1.txt",former_1)
    # np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f2_round_2.txt",former_2)
    # np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f3_round_3.txt",former_3)
    # np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f4_round_4.txt",former_4)
    # np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f5_round_5.txt",former_5)
    # np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f6_round_6.txt",former_6)
    # np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f7_round_7.txt",former_7)
    # np.savetxt("D:/Laser_controller_new_combined/Laser_data/former_data/f8_round_8.txt",former_8)


    
    return former_1, former_2, former_3, former_4, former_5, former_6, former_7, former_8   



    

    
    
    


##################################################################################################################

# former_data_path="D:\\Nisansala\\inference data_31_10\\test_11_10\\former_data"


# fig = plt.figure(figsize=(15, 12))
# ax= fig.add_subplot(111, projection='3d')
# ax.scatter(data_out[:, 0], data_out[:, 1], data_out[:, 2], s=1)

# fig1 = plt.figure(figsize=(15, 12))
# ax1=plt.scatter(data_out[:, 1], data_out[:, 2], marker='.')



######################################################################################################################################