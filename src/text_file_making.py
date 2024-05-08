import os
import glob


folder_path = 'C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\dataset'
output_file_path="C:\\Program Files\\Ansell\\Application\\src_process\\src\\full_data_set\\dataset\\synsetoffset2category.txt"
x=11110000
for batch in range(16):
    mid_folder=folder_path+"\\"+str(x)
    x+=1

    text_files = glob.glob(os.path.join(mid_folder, '*.txt'))
    #print(text_files)
    
    with open(output_file_path, 'a') as output_file:
        for file_path in text_files:
            output_file.write(file_path + '\n')



# Output text file to store file paths
output_file_path = '/path/to/your/output.txt'

# Use glob to get a list of all text files in the folder


# Write the file paths to the output text file


#print(f"File paths written to {output_file_path}")
