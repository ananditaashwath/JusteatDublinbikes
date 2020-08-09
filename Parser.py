'''Author: Anandita Ashwath
    Description:
       Parsing all JSON files from the root directory and converting them into CSV files. 
'''


import os
import json
import numpy as np
import pandas as pd
import time

root_dir="./"
save_csv_dir = './csv'
dir_list =os.listdir(root_dir)
for sub_dir in dir_list:
    child_dir_list = os.listdir(os.path.join(root_dir, sub_dir))
    for child_dir in child_dir_list:
        month_dir_list = os.listdir(os.path.join(root_dir, sub_dir, child_dir))
        for month_dir in month_dir_list:
            save_dir = os.path.join(save_csv_dir, sub_dir, child_dir, month_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_list = os.listdir(os.path.join(root_dir, sub_dir, child_dir, month_dir))
            for file_name in file_list:
                print("processing file: {}".format(os.path.join(root_dir, sub_dir, child_dir, month_dir, file_name)))
                if file_name.endswith('.json'):
                    save_file = os.path.join(save_dir, file_name[:-5] + ".csv")
                    with open(os.path.join(root_dir, sub_dir, child_dir, month_dir, file_name)) as f:
                        data = json.load(f)
                    df = pd.io.json.json_normalize(data)                         
                    df.to_csv(save_file)

print("Finished")
