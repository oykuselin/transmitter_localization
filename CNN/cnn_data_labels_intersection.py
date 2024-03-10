import numpy as np
import os
def read_text_file_as_array(file_path):
    # Read the text file and convert it into a 2D numpy array of floats
    return np.loadtxt(file_path, delimiter=',')
def normalize_to_sphere(coords, radius=5):
    """
    Normalize a point's coordinates so that they lie on the surface of a sphere
    centered at the origin with the specified radius.
    """
    # Convert the coordinates list to a NumPy array for easier calculations
    coords_array = np.array(coords)
    
    # Calculate the norm (distance from the origin) of the original point
    norm = np.linalg.norm(coords_array)
    
    # Calculate the new coordinates, which are the original coordinates
    # normalized to lie on the sphere's surface
    new_coords = (coords_array / norm) * radius
    return new_coords.tolist(), norm

def read_config_for_coordinates_and_labels(config_file_path):
    # Extract coordinates from the config file
    coords = []
    with open(config_file_path, 'r') as file:
        lines = file.readlines()
        # Extract the first three columns of the 6th and 7th rows for transmitter coordinates
        tx1_coords = list(map(float, lines[5].strip().split()[:3]))
        tx2_coords = list(map(float, lines[6].strip().split()[:3]))
        
        # Normalize these points to the sphere's surface and calculate distances
        tx1_normalized, tx1_distance = normalize_to_sphere(tx1_coords)
        tx2_normalized, tx2_distance = normalize_to_sphere(tx2_coords)
        
        # Combine the labels: intersection points followed by distances
        coords.extend(tx1_normalized)
        coords.append(tx1_distance)
        coords.extend(tx2_normalized)
        coords.append(tx2_distance)
    
    return coords

# Update the create_structured_array_with_coords function accordingly:
def create_structured_array_with_coords(file_paths, base_config_path):
    # Updated dtype: we still hold the original data array, but now have 8 label values
    dt = np.dtype([('data', np.float64, (320, 148)), ('labels', np.float64, (8,))])
    structured_array = np.empty(len(file_paths), dtype=dt)
    
    for i, file_path in enumerate(file_paths):
        # Read the text file as a 2D array
        data_array = read_text_file_as_array(file_path)
        
        # Extracting the file number to match with the corresponding config file
        file_number = os.path.basename(file_path).split('_')[-1].split('.')[0]
        config_file_path = os.path.join(base_config_path, f'config_2_{file_number}.txt')
        
        # Read the config file to get new labels
        labels = read_config_for_coordinates_and_labels(config_file_path)
        
        # Assign the 2D array and the new labels to the structured array
        structured_array[i] = (data_array, labels)
    
    return structured_array

# Path to your simulation data and configuration files
simulation_data_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/cnn_data_new/cnn_data_new_neighbors/'
base_config_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/all_simulation_data_2tx/configs/'

file_paths = [os.path.join(simulation_data_path, file) for file in os.listdir(simulation_data_path) if file.endswith('.txt')]

# Create the structured array with data and coordinates
structured_array = create_structured_array_with_coords(file_paths, base_config_path)

# Save the structured array to a .npy file
output_file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/cnn_data_new/data_with_labels_intersection.npy'
np.save(output_file_path, structured_array)

print(f'Structured array with coordinates saved to {output_file_path}')