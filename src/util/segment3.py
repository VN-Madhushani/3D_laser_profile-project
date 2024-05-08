
import numpy as np
import matplotlib.pyplot as plt
import json

class RackSegmentation:
    def __init__(self, size, object_y_ranges):
        self.size = size
        self.object_y_ranges = object_y_ranges

    def segment_points(self, points):
        y = points[:, 1]  # Consider only the y-coordinate
        
        section_ids = np.zeros(len(points), dtype=int)

        for obj_id, y_range in enumerate(self.object_y_ranges):
            lower_bound, upper_bound = y_range
            in_object = np.logical_and(y >= lower_bound, y < upper_bound)
            section_ids[in_object] = obj_id + 1

        return section_ids

    def visualize_rack(self, points):
        section_ids = self.segment_points(points)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=section_ids, cmap='plasma')
        ax.set_title(f"Segmented Rack (Size: {self.size})")
        plt.show()

with open('rack_boundaries.json') as rack_boundaries:
  file_contents =rack_boundaries.read()

# Read 3D data from file
file_path = 'D:/Laser_controller_new_combined/Laser_data/prepared_data/2023_12_01__16_42_13_round_1.txt'
data = np.loadtxt(file_path)  
rack_size = 5


