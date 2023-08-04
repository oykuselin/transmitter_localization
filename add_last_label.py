import os

def process_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        columns = line.strip().split(', ')
        last_column = float(columns[-1])
        second_last_column = float(columns[-2])

        if last_column == 0.0 and second_last_column == 0.0:
            modified_lines.append(line.strip() + ', 1.00000\n')
        else:
            modified_lines.append(line.strip() + ', 0.00000\n')

    with open(file_path, 'w') as f:
        f.writelines(modified_lines)

def main():
    directory = "/Users/berkecaliskan/Documents/transmitter_localization/transmitter_localization/gnn_data_son"  # Replace this with the directory where your txt files are located

    for i in range(1001):  # Assuming you have files from 0 to 1000 (inclusive)
        file_path = os.path.join(directory, f"node_features_{i}.txt")
        if os.path.exists(file_path):
            process_file(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
