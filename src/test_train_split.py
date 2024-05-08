import json
import random

test_train_path="C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\dataset\\train_test_split"
shuffled_test_file_list = test_train_path+"\\test.json"
shuffled_train_file_list = test_train_path+"\\train.json"
shuffled_val_file_list = test_train_path+"\\validation.json"

text_file="C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\dataset\\synsetoffset2category.txt"

with open(text_file, 'r') as file:
    content = file.read()
    lines=content.split('\n')
    random.shuffle(lines)


total_files=16*66
validation_fraction=test_fraction=int(16*66*0.15)
train_fraction=int(16*66*0.7)



with open(shuffled_train_file_list, 'w') as json_file:
    train_file = lines[0:train_fraction]
    json.dump(train_file, json_file)

with open(shuffled_val_file_list, 'w') as json_file:
    val_file = lines[train_fraction:train_fraction+validation_fraction]
    json.dump(val_file, json_file)

with open(shuffled_test_file_list, 'w') as json_file:
    test_file = lines[train_fraction+validation_fraction:train_fraction+validation_fraction+test_fraction]
    json.dump(test_file, json_file)
