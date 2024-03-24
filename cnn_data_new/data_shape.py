import numpy as np

# Assuming 'data_with_labels.npy' is the path to your .npy file
file_path = '/home/oyku/yonsei/new_transmitter_localization/transmitter_localization/cnn_data_new/data_with_labels.npy'  # Replace '/path/to/' with the actual directory path

# Load the NumPy array from the .npy file
array = np.load(file_path)

array_shape = array[0][1].shape
print(array[0][0])
print(array[0][1])
#print(array[1])

num_rows = array_shape[0]
num_columns = array_shape[1] 

print(f"Number of Rows: {num_rows}")
print(f"Number of Columns: {num_columns}")