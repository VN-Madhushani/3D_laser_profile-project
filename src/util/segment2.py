
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




former_y_ranges_rack2 = [
    (0, 260),  # Object 1 (y-coordinate range: 0 to 1)
    (260, 500),  # Object 2 (y-coordinate range: 1 to 2)
    (500, 720),  # Object 3 (y-coordinate range: 2 to 3)
    (720, 970),  # Object 4 (y-coordinate range: 3 to 4)
    (970, 1200),  # Object 5 (y-coordinate range: 4 to 5)
    (1200, 1400),  # Object 6 (y-coordinate range: 5 to 6)
    (1400, 1625),  # Object 7 (y-coordinate range: 6 to 7)
    (1625, 1800),  # Object 8 (y-coordinate range: 7 to 8)
]

former_y_ranges_rack1 = [
    (0, 250),  # Object 1 (y-coordinate range: 0 to 1)
    (250, 450),  # Object 2 (y-coordinate range: 1 to 2)
    (450, 700),  # Object 3 (y-coordinate range: 2 to 3)
    (700, 950),  # Object 4 (y-coordinate range: 3 to 4)
    (950, 1200),  # Object 5 (y-coordinate range: 4 to 5)
    (1200, 1350),  # Object 6 (y-coordinate range: 5 to 6)
    (1350, 1625),  # Object 7 (y-coordinate range: 6 to 7)
    (1625, 1800),  # Object 8 (y-coordinate range: 7 to 8)
]

former_y_ranges_rack3 = [
    (0, 250),  # Object 1 (y-coordinate range: 0 to 1)
    (250, 450),  # Object 2 (y-coordinate range: 1 to 2)
    (450, 700),  # Object 3 (y-coordinate range: 2 to 3)
    (700, 950),  # Object 4 (y-coordinate range: 3 to 4)
    (950, 1200),  # Object 5 (y-coordinate range: 4 to 5)
    (1200, 1350),  # Object 6 (y-coordinate range: 5 to 6)
    (1350, 1625),  # Object 7 (y-coordinate range: 6 to 7)
    (1625, 1800),  # Object 8 (y-coordinate range: 7 to 8)
]

# #2023_12_01__16_44_08_round_1
former_y_ranges_rack4 = [
    (0, 250),  # Object 1 (y-coordinate range: 0 to 1)
    (250, 490),  # Object 2 (y-coordinate range: 1 to 2)
    (490, 700),  # Object 3 (y-coordinate range: 2 to 3)
    (700, 940),  # Object 4 (y-coordinate range: 3 to 4)
    (940, 1150),  # Object 5 (y-coordinate range: 4 to 5)
    (1150, 1375),  # Object 6 (y-coordinate range: 5 to 6)
    (1375, 1625),  # Object 7 (y-coordinate range: 6 to 7)
    (1625, 1800),  # Object 8 (y-coordinate range: 7 to 8)
]

#
former_y_ranges_rack5 = [
    (0, 270),  # Object 1 (y-coordinate range: 0 to 1)
    (270, 500),  # Object 2 (y-coordinate range: 1 to 2)
    (500, 720),  # Object 3 (y-coordinate range: 2 to 3)
    (720, 920),  # Object 4 (y-coordinate range: 3 to 4)
    (920, 1150),  # Object 5 (y-coordinate range: 4 to 5)
    (1150, 1380),  # Object 6 (y-coordinate range: 5 to 6)
    (1380, 1600),  # Object 7 (y-coordinate range: 6 to 7)
    (1600, 1800),  # Object 8 (y-coordinate range: 7 to 8)
]

# Read 3D data from file
file_path = 'D:/Laser_controller_new_combined/Laser_data/prepared_data/2023_12_01__16_42_13_round_1.txt'
data = np.loadtxt(file_path)  
rack_size = 5

if rack_size ==1:
    rack1_segmentation = RackSegmentation(rack_size, former_y_ranges_rack1)
    rack1_segmentation.visualize_rack(data)

elif rack_size == 2:
    rack2_segmentation = RackSegmentation(rack_size, former_y_ranges_rack2)
    rack2_segmentation.visualize_rack(data)

elif rack_size == 3:
    rack2_segmentation = RackSegmentation(rack_size, former_y_ranges_rack3)
    rack2_segmentation.visualize_rack(data)

elif rack_size == 4:
    rack2_segmentation = RackSegmentation(rack_size, former_y_ranges_rack4)
    rack2_segmentation.visualize_rack(data)

else:
    rack5_segmentation = RackSegmentation(rack_size, former_y_ranges_rack5)
    rack5_segmentation.visualize_rack(data)


import json

# Define the former_y_ranges for each rack size
rack_sizes_boundaries = {
    1: former_y_ranges_rack1,
    2: former_y_ranges_rack2,
    3: former_y_ranges_rack3,
    4: former_y_ranges_rack4,
    5: former_y_ranges_rack5,
}

# Save the dictionary to a JSON file
json_file_path = 'D:/Laser_controller_new_combined/Laser_data/rack_boundaries.json'
with open(json_file_path, 'w') as json_file:
    json.dump(rack_sizes_boundaries, json_file)

# Now, you can load the JSON file back into a dictionary
with open(json_file_path, 'r') as json_file:
    loaded_rack_sizes_boundaries = json.load(json_file)

# Access boundaries for a specific rack size
rack_size_to_check = 2
boundaries_for_rack_size = loaded_rack_sizes_boundaries.get(str(rack_size_to_check), [])
print(f"Boundaries for Rack Size {rack_size_to_check}: {boundaries_for_rack_size}")
