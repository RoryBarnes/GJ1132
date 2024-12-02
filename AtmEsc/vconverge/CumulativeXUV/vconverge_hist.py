# import json
# import pandas as pd
# import matplotlib.pyplot as plt

# file=open("output/Converged_Param_Dictionary.json")

# data = json.load(file)
# print(data)

# key = "b,CumulativeXUVFlux,final"
# daCumulativeXUVFlux = data.get(key)

# plt.hist(daCumulativeXUVFlux)
# plt.xlabel('Cumulative XUV Flux [W/m2]')
# plt.ylabel('Number')
# plt.show()

# file.close()
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cumulative_earth_flux = 9.759583e+15

# Step 1: Load and process JSON file
with open("output/Converged_Param_Dictionary.json", 'r') as file:
    content = file.read().strip()  # Read and remove surrounding whitespace
    
    # Check if content is wrapped in double quotes
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1]  # Remove the leading and trailing quotes
        content = content.replace('\\"', '"')  # Replace escaped quotes with normal quotes

    try:
        data = json.loads(content)  # Now load as JSON
        #print("Loaded JSON data:", data)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

# Step 2: Extract the array using the key
key = "b,CumulativeXUVFlux,final"
if isinstance(data, dict):  # Check if the loaded content is a dictionary
    lower_bound = 10
    upper_bound = 1e4
    daCumulativeXUVFlux = data.get(key)
    daCumulativeXUVFlux_clean = pd.Series(daCumulativeXUVFlux).dropna()
    daCumulativeXUVFlux_filtered = np.array(daCumulativeXUVFlux_clean)
    daCumulativeXUVFlux_filtered = daCumulativeXUVFlux_filtered / cumulative_earth_flux

    daCumulativeXUVFlux_filtered = daCumulativeXUVFlux_filtered[(daCumulativeXUVFlux_filtered >= lower_bound) & (daCumulativeXUVFlux_filtered <= upper_bound)]


    #print("Extracted data:", daCumulativeXUVFlux_clean)

    num_bins = 50  # Adjust as needed

    # Define the bin edges in log space
    log_bins = np.logspace(np.log10(min(daCumulativeXUVFlux_filtered)), np.log10(max(daCumulativeXUVFlux_filtered)), num_bins)

    if daCumulativeXUVFlux:
        # Step 3: Plot the data if it's successfully extracted
        plt.hist(daCumulativeXUVFlux_filtered,bins=log_bins)
        plt.xlabel('Cumulative XUV Flux Relative to Modern Earth')
        plt.ylabel('Number')
        plt.xlim(lower_bound,upper_bound)
        plt.xscale('log')
        plt.show()
else:
    print("Loaded data is not a dictionary.")
