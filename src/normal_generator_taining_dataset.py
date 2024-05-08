import open3d as o3d
import numpy as np
import numpy as np
import json
import open3d as o3d
import glob
import h5py
import os
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import time



def normal_generator(point_index, point_cloud):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=150))
    points = np.asarray(point_cloud.points)  
    normal = point_cloud.normals[point_index]
    return normal
    #print("Surface Normal at Point {}: {}".format(point_index, normal))



def is_in_cube(plist,pnt):
    x_max = np.max(plist[:,0])
    x_min = np.min(plist[:,0])
    y_max = np.max(plist[:,1])
    y_min = np.min(plist[:,1])
    z_max = np.max(plist[:,2])
    z_min = np.min(plist[:,2])

    if x_min<= pnt[0] <=x_max and y_min<= pnt[1] <= y_max and z_min<= pnt[2] <= z_max:
        return "in"
    else:
        return "out"


def convert(lwh, point):
    x_1 = x_2 = x_4 = point[0] + lwh[0]/2
    y_1 = y_3 = y_4 = point[1] + lwh[1]/2
    z_1= z_2 = z_3 = point[2] + lwh[2]/2
    z_4 = point[2] - lwh[2]/2
    y_2 = point[1] - lwh[1]/2
    x_3 = point[0] - lwh[0]/2
    arr = [[x_1, y_1, z_1], [x_2, y_2, z_2], [x_3, y_3, z_3], [x_4, y_4, z_4]]
    return arr


def get_json(filename):
    with open(filename) as f:
        data = json.load(f)
    arr = []
    for finger in range(5):
        centroid = np.array([data["objects"][finger]["centroid"]['x'], data["objects"][finger]["centroid"]['y'], data["objects"][finger]["centroid"]['z']])
        lwh = np.array([data["objects"][finger]["dimensions"]['length'], data["objects"][finger]["dimensions"]['width'], data["objects"][finger]["dimensions"]['height']])
        #arr.append([lwh, centroid])
        arr.append(convert(lwh,centroid))
    arr = np.array(arr)
    #print(arr.shape)
    return arr



##################################################################################################

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



###################################################################################################
# train_hdf5_list = open("D:\\UoP\\pointnet-master\\ply_json", "w")
# output_directory = os.path.dirname(train_hdf5_list)
# os.makedirs(output_directory, exist_ok=True)


directory = "C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\ply_json"
synsetoffset_2_category = []

x_init=11110000
count=0

ply_files=glob.glob(directory+'/*ply')
np.random.shuffle(ply_files)
for index in range(66*16):
    filename=ply_files[index]
    pcd = o3d.io.read_point_cloud(filename)
    print(type(pcd))
    break
    point_arr = np.asarray(pcd.points)
    #print(max(cord[0] for cord in point_arr))
    point_arr = downsample(point_arr)
    normals = np.zeros(((point_arr.shape)[0],3))
    indices = np.zeros(((point_arr.shape)[0],1)).astype(int)
    # print(point_arr.shape)
    # print(normals.shape)
    # print(indices.shape)
    
    # fig1 = plt.figure(figsize=(15, 12))
    # ax2= fig1.add_subplot(111, projection='3d')
    # ax2.scatter(point_arr[:, 0], point_arr[:, 1], point_arr[:, 2], s=1)
    # plt.show()

    file_name = (filename.split("\\")[-1]).split(".")[0]
    print(filename)
    json_filename = directory +"\\"+ file_name + '.json'
    json_file = open(json_filename)
    json_arr = get_json(json_filename)
    #print(len(point_arr))


    x1 = json_arr[0,:,:]
    x2 = json_arr[1,:,:]
    x3 = json_arr[2,:,:]
    x4 = json_arr[3,:,:]
    x5 = json_arr[4,:,:]

    for index in range(0,len(point_arr)):
        #print(point_arr[index])
        

        if is_in_cube(x1,point_arr[index])=="in":
            indices[index,:] = int(1)
        elif is_in_cube(x2,point_arr[index])=="in":
            indices[index,:] = int(2)
        elif is_in_cube(x3,point_arr[index])=="in":
            indices[index,:] = int(3)
        elif is_in_cube(x4,point_arr[index])=="in":
            indices[index,:] = int(4)
        elif is_in_cube(x5,point_arr[index])=="in":
            indices[index,:] = int(5)
        else:
            continue
        
        normals[index,:] = normal_generator(index,pcd)

    
    final_array = np.hstack((point_arr,normals,indices))
    print(final_array.shape)


    quotient=count//66
    x=x_init+quotient
    count+=1
    
    #print(x)
    path="C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\dataset"
    


    folder_path=path+"\\" + str(x)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass
        #print(f"Folder '{folder_path}' already exists.")

    txt_filename = folder_path +"\\"+ file_name +".txt"
    np.savetxt(txt_filename, final_array, delimiter = " ")

    
    # train_hdf5_list.write(h5_filename)
    synsetoffset_2_category.append(folder_path+"\\"+ file_name)

#synsetoffset_2_category = np.array(synsetoffset_2_category)
#print("synsetoffset_2_category", synsetoffset_2_category.shape)

test_train_path="C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\dataset\\train_test_split"
shuffled_test_file_list = test_train_path+"\\test.json"
shuffled_train_file_list = test_train_path+"\\train.json"
shuffled_val_file_list = test_train_path+"\\validation.json"

synsetoffset2category = np.array(synsetoffset_2_category)
np.random.shuffle(synsetoffset2category)
path="C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\dataset"
np.savetxt(path+"\\synsetoffset2category.txt", synsetoffset2category, fmt = "%s")
      
total_files=16*66
validation_fraction=test_fraction=int(16*66*0.15)
train_fraction=int(16*66*0.7)

with open(shuffled_train_file_list, 'w') as json_file:
    train_file = synsetoffset_2_category[0:train_fraction]
    json.dump(train_file, json_file)

with open(shuffled_val_file_list, 'w') as json_file:
    val_file = synsetoffset_2_category[train_fraction:train_fraction+validation_fraction]
    json.dump(val_file, json_file)

with open(shuffled_test_file_list, 'w') as json_file:
    test_file = synsetoffset_2_category[train_fraction+validation_fraction:train_fraction+validation_fraction+test_fraction]
    json.dump(test_file, json_file)


