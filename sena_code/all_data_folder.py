import os

current_dir = os.getcwd()
parent_directory = os.path.dirname(current_dir)


num_of_current_simulations = len(os.listdir(os.path.join(current_dir, "all_simulation_data_2tx/configs")))

for i in range(len(os.listdir(os.path.join(current_dir, "data_augmented_rotation/configs")))):
    config_lines = []
    result_lines = []
    with open(os.path.join(current_dir, "data_augmented_rotation/configs/config_2_aug_{}.txt".format(i)), "r") as file:
        config_lines = file.readlines()
    with open(os.path.join(current_dir, "data_augmented_rotation/results/result_2_aug_{}.txt".format(i)), "r") as file:
        result_lines = file.readlines()

    config_filename = "config_2_{}".format(i + num_of_current_simulations)
    result_filename = "result_2_{}".format(i + num_of_current_simulations)

    with open(os.path.join(current_dir, "all_simulation_data_2tx/configs/{}.txt".format(config_filename)), "a") as file:
        file.writelines(config_lines)

    with open(os.path.join(current_dir, "all_simulation_data_2tx/results/{}.txt".format(result_filename)), "a") as file:
        file.writelines(result_lines)

    print(i)