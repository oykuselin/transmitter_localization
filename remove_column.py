import os

def process_file(read_file_path,write_file_path):
    with open(read_file_path, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        columns = line.strip().split(', ')
        modified_columns = columns[:-2]  # Exclude the last column
        modified_line = ', '.join(modified_columns) + '\n'
        modified_lines.append(modified_line)

    with open(write_file_path, 'w') as f:
        f.writelines(modified_lines)

def main():
    current_path = os.getcwd()
    subdir = 'gnn_data_son'
    read_directory = os.path.join(current_path, subdir)
    write_directory = "/home/oyku/yonsei/new_transmitter_localization/transmitter_localization/cnn_data_new"


    for i in os.listdir(read_directory):  # Assuming you have files from 0 to 1000 (inclusive)
        read_file_path = os.path.join(read_directory, i)
        write_file_path = os.path.join(write_directory, i)
        if os.path.exists(read_file_path):
            process_file(read_file_path,write_file_path)
        else:
            print(f"File not found: {read_file_path}")

if __name__ == "__main__":
    main()