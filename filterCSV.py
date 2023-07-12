import pandas as pd

# Load the original CSV file
data = pd.read_csv('flow_statistics5.csv')

# Define the flow_id to filter
flow_id = 32088147978799629  # Replace with the desired flow_id

# Filter the data based on the flow_id
filtered_data = data[data['Flow_Id'] == flow_id]

# Specify the file name for the filtered data
filtered_file_name = 'filtered_data.csv'  # Replace with the desired file name

# Save the filtered data to a new CSV file
filtered_data.to_csv(filtered_file_name, index=False)
