import numpy as np
import os

def read_text_file_as_array(file_path):
    # Read the text file and convert it into a 2D numpy array of floats
    return np.loadtxt(file_path, delimiter=',')

def read_config_for_coordinates(config_file_path):
    # Extract coordinates from the config file
    coords = []
    with open(config_file_path, 'r') as file:
        lines = file.readlines()
        # Extract the first three columns of the 6th and 7th rows for transmitter coordinates
        coords.extend(lines[5].strip().split()[:3])  # 6th row, first 3 columns
        coords.extend(lines[6].strip().split()[:3])  # 7th row, first 3 columns
    # Convert list of string coords to float
    coords = [float(coord) for coord in coords]
    return coords

def create_structured_array_with_coords(file_paths, base_config_path):
    # Define the dtype for the structured array to hold a (320, 148) float array and a (6,) float array for coords
    dt = np.dtype([('data', np.float64, (320, 148)), ('coords', np.float64, (6,))])
    # Initialize an empty array with the defined dtype
    structured_array = np.empty(len(file_paths), dtype=dt)
    
    count  =  0

    for i, file_path in enumerate(file_paths):
        # Read the text file as a 2D array
        data_array = read_text_file_as_array(file_path)
        
        # Extracting the file number to match with the corresponding config file
        file_number = os.path.basename(file_path).split('_')[-1].split('.')[0]
        config_file_path = os.path.join(base_config_path, f'config_2_{file_number}.txt')
        
        if count < 3:
            print(config_file_path)
        count = count + 1
        # Read the config file to get coordinates
        coords = read_config_for_coordinates(config_file_path)
        
        # Assign the 2D array and the coordinates to the structured array
        structured_array[i] = (data_array, coords)
    
    return structured_array

# Path to your simulation data and configuration files
simulation_data_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/cnn_data_new/cnn_data_new_neighbors/'
base_config_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/all_simulation_data_2tx/configs/'

file_paths = [os.path.join(simulation_data_path, file) for file in os.listdir(simulation_data_path) if file.endswith('.txt')]

# Create the structured array with data and coordinates
structured_array = create_structured_array_with_coords(file_paths, base_config_path)

# Save the structured array to a .npy file
output_file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/cnn_data_new/data_with_labels.npy'
np.save(output_file_path, structured_array)

print(f'Structured array with coordinates saved to {output_file_path}')
