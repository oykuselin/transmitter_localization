import os
from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import random


def rotate(input, roll, pitch, yaw):

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    R = np.array([[cos_yaw*cos_pitch, cos_yaw*sin_pitch*sin_roll - sin_yaw*cos_roll, cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll],
                  [sin_yaw*cos_pitch, sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll, sin_yaw*sin_pitch*cos_roll - cos_yaw*sin_roll],
                  [-sin_pitch, cos_pitch*sin_roll, cos_pitch*cos_roll]])
    
    return (R @ input)

def rotate_rod(vector, angle, axis):

    axis = axis / np.linalg.norm(axis)
    
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    rotated_vector = vector * cos_theta + np.cross(axis, vector) * sin_theta + axis * np.dot(axis, vector) * (1 - cos_theta)
    
    return rotated_vector 

def get_mol_locations(mixed, tx_id):
    return mixed[mixed[:, 4] == tx_id]

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
config_sub_directory = 'test/2tx/configs'
results_sub_directory = 'test/2tx/results'

configs_dir = os.path.join(current_directory, config_sub_directory)
results_dir = os.path.join(current_directory, results_sub_directory)

configs_file_list = os.listdir(configs_dir)
results_file_list = os.listdir(results_dir)

if not os.path.exists("data_augmented_rotation"):
    os.makedirs("data_augmented_rotation")
    os.makedirs("data_augmented_rotation/configs")
    os.makedirs("data_augmented_rotation/results")

number_of_aug_current = len(os.listdir(os.path.join(current_directory, "data_augmented_rotation/configs")))
AUG_NUM = 5

for i in range(AUG_NUM):

    aug_conf_lines = []
    aug_result_lines = []
    track_list = []
    for j in range(2):
        random_config_file = np.random.choice(configs_file_list)

        match = re.search(r"_\d+(\.)", random_config_file)
        if match:
            exp_number = match.group(0)[1:-1]

        with open(os.path.join(configs_dir, random_config_file), "r") as file:
            lines = file.readlines()
            tx_id = random.randint(0,1)
            random_tx_line = lines[5 + tx_id]

        aug_conf_lines.append(random_tx_line)

        result_file_name = 'result_2_{}.txt'.format(exp_number)
        print(result_file_name)
        mixed_data = np.loadtxt(os.path.join(results_dir, result_file_name))
        
        tx_data = get_mol_locations(mixed_data, tx_id)
        coordinates = tx_data[:, :3]
        rest = (tx_data[:, 3:]).astype(int)
        
        ROLL = np.deg2rad(random.uniform(0, 360)) # 0 VE 360 I DAHIL ETMELI MI ???
        PITCH = np.deg2rad(random.uniform(0, 360))
        YAW = np.deg2rad(random.uniform(0, 360))
        rotated_points = np.apply_along_axis(rotate, 1, coordinates, ROLL, PITCH, YAW)
        aug_tx_data = np.concatenate((rotated_points, rest), axis=1)
        aug_tx_data[:, 4] = j # !!!!!!
        aug_result_lines.append(aug_tx_data)

        track_list.append("{}th tx is from the experiment {} with the tx_id {}\nrotation angles roll: {}, pitch: {}, yaw: {}\n".format(j, exp_number, tx_id, ROLL, PITCH, YAW))

    with open("track.txt", "a") as file:
        file.writelines(track_list)

    # create new config file for the augmented data
    with open(os.path.join(current_directory, 'data_augmented_rotation/configs/config_2_aug_{}.txt'.format(i + number_of_aug_current)), "a") as file:
        file.write("1e-06\n5\n79.4\n0.5\n2\n")
        file.writelines(aug_conf_lines)
        file.write("1\n0 0 0")

    new_result = np.vstack((aug_result_lines[0], aug_result_lines[1]))
    np.savetxt('data_augmented_rotation/results/result_2_aug_{}.txt'.format(i + number_of_aug_current), new_result, delimiter=' ', fmt=' '.join(['%1.6f']*3 + ['%i']*2))


    
        

# ROLL = np.deg2rad(0) # must be a random number ]0,360[
# PITCH = np.deg2rad(0) # must be a random number ]0,360[
# YAW = np.deg2rad(0) # must be a random number ]0,360[

# # ANGLE = np.deg2rad(0)
# # AXIS = [1, 0, 0] 


# data = np.loadtxt("/Users/senamumcu/Desktop/2020_MultiTxLocalization/3tx_1/result1/0.txt")
# x, y, z = data[:, 0], data[:, 1], data[:, 2]

# original_points = np.column_stack((x,y,z))
# rotated_points = np.apply_along_axis(rotate, 1, original_points, "sena")
# rotated_points_spherical = np.apply_along_axis(rotate_rod, 1, original_points)

# # np.savetxt('cartesion_rotation.txt', rotated_points, delimiter=', ')
# # np.savetxt('spherical_rotation.txt', rotated_points_spherical, delimiter=', ')


# receiver_x = 0  
# receiver_y = 0
# receiver_z = 0 

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c='green', label='original')
# ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], c='blue', label='rot1')
# ax.scatter(rotated_points_spherical[:, 0], rotated_points_spherical[:, 1], rotated_points_spherical[:, 2], c='red', label='rot2')
# ax.scatter(receiver_x, receiver_y, receiver_z, c='black', s=100, label='Receiver')
# plt.show()
