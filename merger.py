import os
import pandas as pd

# Define the folder name
folder_name = 'results'

# Define the file names
file_names = ['SFA', 'VNA', 'SHA', 'SYA', 'DFA']

# Create an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Iterate over the file names and merge the data
for file_name in file_names:
    # Construct the file path for binary file
    binary_file_path = os.path.join(folder_name, file_name)

    # Read the binary file into a DataFrame
    df = pd.read_csv(binary_file_path, sep=';', header=None)

    # Add a column to identify the source file
    df['Source'] = file_name

    # Append the data to the merged_data DataFrame
    merged_data = pd.concat([merged_data, df], ignore_index=True)

# Save the merged data to a new CSV file
merged_data.to_csv('merged_data.csv', index=False)
