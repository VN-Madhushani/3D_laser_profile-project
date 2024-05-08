import keyboard
import time
#import online_data_gathering_copy
import online_data_gathering_buffer_line
from datetime import datetime
from get_round_number import get_round

date = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
current_round = get_round(date)


print(f"{current_round} started")

print("waiting for data grabbing round {}...".format(current_round))

print("data grabbing.....")

# prepared_data = online_data_gathering_copy.gathering_data(date,current_round)
prepared_data = online_data_gathering_buffer_line.gathering_data(date,current_round)

print("============")
print("============")

current_round +=1
print("waiting for data grabbing round {}...".format(current_round))

time.sleep(0.0001)




    
