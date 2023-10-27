import numpy as np
import os

all_sim_data = []

for i in range(1):
    path = "/Users/senamumcu/Desktop/2020_MultiTxLocalization/kerem's data{}/results".format(i)
    for j in range(3): #len(os.listdir(path))):
        file_path = os.path.join(path, "result_2_{}.txt".format(j + i * 5000))
        # print(file_path)
        simulation = np.loadtxt(file_path, delimiter=" ")
        all_sim_data.append(simulation)

# print(len(all_sim_data))
# np.save("all_simulations.npy", all_sim_data)
