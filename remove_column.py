import os

def process_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        columns = line.strip().split(', ')
        modified_columns = columns[:-1]  # Exclude the last column
        modified_line = ', '.join(modified_columns) + '\n'
        modified_lines.append(modified_line)

    with open(file_path, 'w') as f:
        f.writelines(modified_lines)

def main():
    current_path = os.getcwd()
    subdir = 'cnn_data_final'
    directory = os.path.join(current_path, subdir)

    for i in range(1001):  # Assuming you have files from 0 to 1000 (inclusive)
        file_path = os.path.join(directory, f"node_features_{i}.txt")
        if os.path.exists(file_path):
            process_file(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()