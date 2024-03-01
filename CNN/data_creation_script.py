import os 
import numpy as np

curr_dir = os.getcwd()

data_list = []
for i in range(1000):
    data = np.loadtxt("{}/CNN/cnn_data_final/node_features_{}.txt".format(curr_dir, i), delimiter=",")
    data = np.loadtxt("{}/CNN/cnn_data_final/node_features_{}.txt".format(curr_dir, i), delimiter=", ")
    data = data.reshape(-1, 320, 151)
    data_list.append(data)

data_list = np.asarray(data_list)
np.save("CNN/Data_3label.npy", data_list)