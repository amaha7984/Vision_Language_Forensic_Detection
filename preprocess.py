import os
import random
import numpy as np
from tqdm import tqdm

data_root_train = './data/Train'
data_root_test = './data/Test'
save_path = 'annotation/' 

def FileList_Generation(data_root, save_filename, num_files=None):
    data_name_list = [
        ('Fake/MapGen', 1),  # Fake Dataset
        ('Real/MapReal', 0),  # Real Dataset
    ]

    img_list = []
    img_list2 = []
    
    for data_name, label in data_name_list:
        path = f'{data_root}/{data_name}/'
        flist = sorted(os.listdir(path))
        if num_files:
            flist = flist[:num_files] 

        for file in tqdm(flist, desc=f'Processing {data_name}'):
            img_list.append((data_name + '/' + file, label))
    
    print(f'#Images: {len(img_list)}')
    
    textfile = open(save_path + save_filename, 'w')
    for item in img_list:
        textfile.write(f'{item[0]} {item[1]}\n')
    textfile.close()

if __name__ == '__main__':
    # Create train and test file lists
    FileList_Generation(data_root_train, 'Train.txt')
    FileList_Generation(data_root_test, 'Test.txt')
