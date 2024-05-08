import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import curve_fit

def curve_xy(cloud):
    cloud = cloud[cloud[:,1].argsort()]
    arr = []
    init_y = -99
    for point in cloud:
        if init_y!= point[1]:
            A = [item for item in cloud if item[1] == point[1]]
            A = np.array(A)
            A = A[A[:,0].argsort()]
            arr.append(A[0])
            init_y = point[1]
        else: continue
    arr = np.array(arr)
    #print("arr: ", arr.shape)
    return arr

def z_outlier(cloud, threshold = 3):
    mean = np.mean(cloud[:, 1])
    std = np.std(cloud[:, 1])
    outliers = []
    for i in range(len(cloud)):
        z = (float(cloud[i, 1]) - mean) / std
        if np.abs(z) < threshold:
            outliers.append(i)
    return np.array(cloud[outliers])


def minY(cloud_finger):
    cloud_xy = curve_xy(cloud_finger)
    cloud_xy = z_outlier(cloud_xy)
    y = cloud_xy[:, 0]
    x = cloud_xy[:, 1]
    plt.scatter(x, y, s=1)
    plt.title('A Basic Scatter Plot')
    plt.xlabel('X-axis') 
    plt.ylabel('Y-axis')
    plt.show()
    ######################################################
    pts_xy = np.transpose(cloud_xy[:,0:2])
    #print(pts_xy.shape)
    (tck, u), fp, ier, msg = splprep(pts_xy, u=None, per=0, k=5, full_output=True) #s = optional parameter (default used here)
    #print('Spline score:',fp) #goodness of fit flatlines after a given s value (and higher), which captures the default s-value as well 
    x_new, y_new = splev(u, tck, der=0)
    print(tck)
    x_new, y_new = y_new, x_new
    coefficient = np.polyfit(x, y, 2)
    [a, b, c] = coefficient
    coefficient_ = np.polyfit(x_new, y_new, 2)
    [a_, b_, c_] = coefficient_
    x_min, y_min = -b_/(2*a_), c_-b_**2/(4*a_)
    #y_raw = coefficient[0]*(x**2) + coefficient[1]*x + coefficient[2]
    y_raw = np.poly1d(coefficient)(x)
    y_spl = coefficient_[0]*(x_new**2) + coefficient_[1]*x_new + coefficient_[2]
    plt.scatter(x, y, s=1)
    plt.plot(x_new, y_new, 'k')
    plt.plot(x, y_raw, 'red')
    plt.plot(x_new, y_spl, 'blue')
    plt.vlines(x_min, c_-b_**2/(4*a_), c_-b_**2/(4*a_)+10, 'green')
    plt.scatter(x_min, y_min, s=16, color = 'k')
    plt.show()