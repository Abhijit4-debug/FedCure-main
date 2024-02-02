import numpy as np
import os
import random
from utils.HAR_utils import *
import numpy as np
import pandas as pd
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
data_path = "har/"
dir_path = "har_practical_non_iid_unbalanced/"
num_clients=18
num_classes=6
data_folder=data_path+'rawdata/'
str_folder = data_folder + 'UCI HAR Dataset/'

def generate_har(dir_path, num_clients, num_classes, niid, balance, partition):
    INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"


    str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                        INPUT_SIGNAL_TYPES]
    str_test_files = [str_folder + 'test/' + 'Inertial Signals/' +
                        item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
    str_train_y = str_folder + 'train/y_train.txt'
    str_test_y = str_folder + 'test/y_test.txt'
    str_train_id = str_folder + 'train/subject_train.txt'
    str_test_id = str_folder + 'test/subject_test.txt'

    X_train = format_data_x(str_train_files)
    X_test = format_data_x(str_test_files)
    Y_train = format_data_y(str_train_y)
    Y_test = format_data_y(str_test_y)
   

    X_train, X_test = X_train.reshape((-1, 9, 1, 128)), X_test.reshape((-1, 9, 1, 128))

    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)
    import torch
    X=torch.tensor(X)
    Y=torch.tensor(Y)

    X, y, statistic = separate_data((X, Y), num_clients, num_classes, 
                                        niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
            statistic, niid, balance, partition)
    
if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_har(dir_path, num_clients, num_classes, niid, balance, partition)