import os
import re
import glob
import numpy as np

# Initialize an array to store the relative energy differences
energy_relative_differences = []

# Iterate through matching directories
for subdir in sorted(glob.glob("output/stabilityrand_*")):
    # Construct the full path to the log file
    log_file_path = os.path.join(subdir, "gj1132.log")
    
    # Check if file exists
    if not os.path.exists(log_file_path):
        print(f"File not found: {log_file_path}")
        continue
    
    try:
        # Read the file
        with open(log_file_path, 'r') as file:
            # Lists to store energy values
            energy_values = []
            
            # Iterate through lines
            for line in file:
                # Look for the specific energy line
                if "(TotEnergy) Total System Energy [kg*m^2/sec^2]:" in line:
                    # Extract the floating point value
                    match = re.search(r': ([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)', line)
                    if match:
                        energy_values.append(float(match.group(1)))
            
            # Check if we found at least two energy values
            if len(energy_values) >= 2:
                # Calculate relative difference
                relative_diff = abs((energy_values[-1] - energy_values[0]) / energy_values[0])
                energy_relative_differences.append(relative_diff)
            else:
                print(f"Insufficient energy values in {log_file_path}")
    
    except IOError as e:
        print(f"Error reading {log_file_path}: {e}")

# Convert to numpy array for further processing
energy_relative_differences = np.array(energy_relative_differences)

# Print the results
print(f"Total number of relative differences calculated: {len(energy_relative_differences)}")
print("First few relative differences:", energy_relative_differences[:5])

# Save to a file
np.savetxt('energy_relative_differences.txt', energy_relative_differences)


