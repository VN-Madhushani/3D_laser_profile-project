import json


# cat={}
# catfile="C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\dataset\\formerclasses.txt"


# with open(catfile, 'r') as f:
#     for line in f:
#         ls = line.strip().split()
#         cat[ls[0]] = ls[1]
#         #print(line)
#         #print(ls)
        
#         #print(cat)

# cat = {k: v for k, v in cat.items()}
# print(cat)
'''
file="C:\\Program Files\\Ansell\\Application\\src_process\src\\full_data_set\dataset\\train_test_split\\test.json"
with open(file, 'r') as f:
    train_ids = set([str(d.split('\\')[6]) for d in json.load(f)])
'''

# a=[10,0,20]
# a.sort()
# print(a)


# import json

# file_path='C:\\Program Files\Ansell\\Application\\src_process\\src\\full_data_set\dataset\\train_test_split\\validation.json'




# with open(file_path, 'r') as json_file:
#     # Load the JSON data
#     data = json.load(json_file)

#     # Count the number of objects (items)
#     num_objects = len(data)

# # Print the result
# print(f'Number of objects in the JSON file: {num_objects}')

with open("C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\dataset\\train_test_split\\test.json", 'r') as f:
            test_ids = set([str(d.split('\\')[-2]) for d in json.load(f)])

print(test_ids)