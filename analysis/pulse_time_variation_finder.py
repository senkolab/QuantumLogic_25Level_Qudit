# import os

# from datetime import datetime

# import numpy as np
# import pandas as pd

# # Define the directory containing the .txt files
# directory = '../data/pulse_time_calibrations_data'

# # Initialize lists to hold data
# datetime_list = []
# first_column_data = []
# second_column_data = []

# # Loop through each file in the directory
# for filename in os.listdir(directory):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(directory, filename)

#         # Check if the file has exactly 7 lines
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#             if len(lines) == 7:
#                 # Extract the datetime from the filename
#                 file_datetime = filename.split('_')[-2] + '_' + filename.split('_')[-1][:4]  # YYYYMMDD HHMM
#                 print(file_datetime)
#                 dt = datetime.strptime(file_datetime, '%Y%m%d_%H%M')
#                 datetime_list.append(dt)

#                 # Extract values from each line
#                 for line in lines:
#                     parts = line.split(',')
#                     first_column_data.append(float(parts[0].strip()))  # First column
#                     second_column_data.append(float(parts[1].strip()))  # Second column
#                 print(np.shape(first_column_data))
#                 print(np.shape(second_column_data))

# # Create DataFrames for first and second columns
# df_first_col = pd.DataFrame({
#     'Datetime': datetime_list,
#     'First Column': first_column_data
# })

# df_second_col = pd.DataFrame({
#     'Datetime': datetime_list,
#     'Second Column': second_column_data
# })
# breakpoint()

# # Saving the dataframes to CSV files
# df_first_col.to_csv('first_column_data.csv', index=False)
# df_second_col.to_csv('second_column_data.csv', index=False)

# print("Data extraction complete. Data saved to CSV files.")

import os

import numpy as np

# Define the directory containing the .txt files
directory = '../data/pulse_time_variations_data'

# Initialize an array to hold the center column data
referenced_pulse_time_data = []

# Loop through each file in the directory
center_column_data = []
for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # Only process .txt files
        file_path = os.path.join(directory, filename)

        # Open the file and read its lines
        with open(file_path, 'r') as file:
            lines = file.readlines()

            # Process the lines starting from the second row (index 1)
            for line in lines[2:]:  # Skip the first two rows
                # Split the line, extract the center column value and strip whitespace/newlines
                parts = line.split(',')
                if len(parts
                       ) >= 2:  # Ensure there are enough parts in the line
                    center_value = float(parts[1].strip())

                    # Append the center value to the array
                    center_column_data.append(center_value)

file_data = np.array(center_column_data)
percent_error_data = np.abs(file_data - file_data[0]) / file_data[0]
referenced_pulse_time_data.append(percent_error_data[1:])

# Output the collected data
pulse_time_variations = np.concatenate(referenced_pulse_time_data)
pulse_time_std = np.std(pulse_time_variations)
print('The long-term (~few hours) time scale variation in pulse times is ',
      np.round(100*pulse_time_std,5), '%.')
