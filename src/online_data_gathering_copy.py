import ctypes
import mmap
import time
import numpy as np
import json
import matplotlib.pyplot as plt

# BUFFER_LINE_COUNT = 325
# LINE_PIXEL_COUNT = 2120
#BLC = 400    LPC = 2120 ACQ = 3
BUFFER_LINE_COUNT = 600
LINE_PIXEL_COUNT = 2120
ACQ_NUM_BUFFERS = 3


class TD(ctypes.Structure):
        _fields_ = [
            ('current_buffer', ctypes.c_int),
            ('line_no_buffer_a', ctypes.c_int),
            ('line_no_buffer_b', ctypes.c_int),
            ('line_data_buffer_a', (ctypes.c_int * LINE_PIXEL_COUNT) * BUFFER_LINE_COUNT),
            ('line_data_buffer_b', (ctypes.c_int * LINE_PIXEL_COUNT) * BUFFER_LINE_COUNT),
            ('is_reading', ctypes.c_bool),
            ('started',ctypes.c_bool),
            ('is_buffer_a_write_finished', ctypes.c_bool),
            ('is_buffer_b_write_finished', ctypes.c_bool),
            ('is_buffer_a_busy_reading', ctypes.c_bool),
            ('is_buffer_b_busy_reading', ctypes.c_bool)
        ]

#def gathering_data(): 
def gathering_data(date, round,save_files=True):
    print("done b")
    shmem = mmap.mmap(-1, ctypes.sizeof(TD), "TD75")
    i = 0
    data = TD.from_buffer(shmem)

    # Start Acquisition
    data.is_reading = True

    np_data_all = np.empty((0, LINE_PIXEL_COUNT), int)
    #np_data_all1 = np.empty((0, LINE_PIXEL_COUNT), int)

    total_buffer_count = 0
    current_buffer = 0
    #################################################
    #data.is_buffer_a_write_finished = True
    #data.is_buffer_b_write_finished = True
    ##############################################
    

    
    while total_buffer_count < ACQ_NUM_BUFFERS:
        #print(data.is_buffer_a_write_finished)

        if data.is_buffer_a_write_finished and current_buffer==0:
            data.is_buffer_a_busy_reading = True
            print('Start A')
            np_buffer_a = np.ctypeslib.as_array(data.line_data_buffer_a)
            np_data_all = np.append(np_data_all, np_buffer_a, axis=0)
            print(np_buffer_a.shape)
            print(np_data_all.shape)
            total_buffer_count += 1
            print('End A')
            data.is_buffer_a_busy_reading = False
            current_buffer = 1
            data.started=True
            
        elif data.is_buffer_b_write_finished and current_buffer == 1:
            data.is_buffer_b_busy_reading = True
            print('Start B')
            np_buffer_b = np.ctypeslib.as_array(data.line_data_buffer_b)
            np_data_all = np.append(np_data_all, np_buffer_b, axis=0)
            print(np_buffer_b.shape)
            print(np_data_all.shape)
            total_buffer_count += 1
            print('End B')
            data.is_buffer_b_busy_reading = False
            current_buffer = 0
        time.sleep(0.0001)
        

    # print("done d")
    #print(np_data_all)
    print(np_data_all.shape)
    print("done e")
    data.is_reading = False

 
    y_data = np.array([x for (x, y), element in np.ndenumerate(np_data_all) if (0<element<180000 and 0<x<1700) ])
    z_data = np.array([element for (x, y), element in np.ndenumerate(np_data_all)  if (0<element<180000 and 0<x<1700)] )
    x_data = np.array([y for (x, y), element in np.ndenumerate(np_data_all)  if (0<element<180000 and 0<x<1700)])
    # print(len(y_data))
    # print(len(z_data))



        
    z_data= z_data *(-1)
    a = len(z_data)
        
    # fig = plt.figure(figsize=(15, 12))

    delind = [ i  for i in range(0, a) if i % 20 != 0]
    
    y_data= np.delete(y_data,delind)
    x_data= np.delete(x_data,delind)
    z_data= np.delete(z_data,delind)

    data_out = np.array([x_data, y_data, z_data])
    data_out = data_out.transpose()

    #arrange data in ascending y order
    index=1
    data_out = data_out[data_out[:,index].argsort()]


    

    

    # if save_files:
    #     np.savetxt("D:/Nisansala/Laser_profiler_operation_finalized/Laser_controller/raw_data/{}_round_{}_x_data.txt".format(date,round),x_data)
    #     np.savetxt("D:/Nisansala/Laser_profiler_operation_finalized/Laser_controller/raw_data/{}_round_{}_y_data.txt".format(date,round),y_data)
    #     np.savetxt("D:/Nisansala/Laser_profiler_operation_finalized/Laser_controller/raw_data/{}_round_{}_z_data.txt".format(date,round),z_data)
    #     np.savetxt("D:/Nisansala/Laser_profiler_operation_finalized/Laser_controller/raw_data/{}_round_{}_all_data.txt".format(date,round),np_data_all)
    #     np.savetxt("D:/Nisansala/Laser_profiler_operation_finalized/Laser_controller/prepared_data/{}_round_{}.txt".format(date,round),data_out)
    rack_size='1R'
    if save_files:
        np.savetxt("D:/Laser_controller_new_combined/new_Data/raw_data/{}_round_{}_x_data_{}.txt".format(date,round,rack_size),x_data)
        np.savetxt("D:/Laser_controller_new_combined/new_Data/raw_data/{}_round_{}_y_data_{}.txt".format(date,round,rack_size),y_data)
        np.savetxt("D:/Laser_controller_new_combined/new_Data/raw_data/{}_round_{}_z_data_{}.txt".format(date,round,rack_size),z_data)
        np.savetxt("D:/Laser_controller_new_combined/new_Data/raw_data/{}_round_{}_all_data_{}.txt".format(date,round,rack_size),np_data_all)
        np.savetxt("D:/Laser_controller_new_combined/new_Data/prepared_data_demo/{}_round_{}_{}.txt".format(date,round,rack_size),data_out)

    # if save_files:
    #     np.savetxt("D:/Laser_controller_new_combined/new_Data/raw_data/{}_round_{}_x_data_5L.txt".format(date,round),x_data)
    #     np.savetxt("D:/Laser_controller_new_combined/new_Data/raw_data/{}_round_{}_y_data_5L.txt".format(date,round),y_data)
    #     np.savetxt("D:/Laser_controller_new_combined/new_Data/raw_data/{}_round_{}_z_data_5L.txt".format(date,round),z_data)
    #     np.savetxt("D:/Laser_controller_new_combined/new_Data/raw_data/{}_round_{}_all_data_5L.txt".format(date,round),np_data_all)
    #     np.savetxt("D:/Laser_controller_new_combined/new_Data/prepared_data/{}_round_{}_5L.txt".format(date,round),data_out)


# 11_22 to optimize time efficiency
    #plotting prepared data    
    # ax= fig.add_subplot(111, projection='3d')
    # ax.scatter(data_out[:, 0], data_out[:, 1], data_out[:, 2], s=1)
    fig = plt.figure()

    # ax= fig.add_subplot(111)

    cloud1=data_out
    ax= fig.add_subplot(111)
    # ax.scatter(cloud1[:, 0], cloud1[:, 1], c='blue', marker='o', label='Cloud 1')
    ax.scatter(cloud1[:, 0], cloud1[:, 1], c='blue', marker='o', label='Cloud 1')
    
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


    return data_out