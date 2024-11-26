# import json
# import pandas as pd
# import matplotlib.pyplot as plt

# file=open("500TO/Converged_Param_Dictionary.json")

# data = json.load(file)

# daWaterFinal=data["b,SurfWaterMass,final"]

# #print(daWaterFinal)
# #print(daWaterFinal[0])

# plt.hist(daWaterFinal)
# plt.xlabel('Final Water (TO)')
# plt.ylabel('Number')
# plt.show()

# file.close()

import json
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load and process JSON file
with open("output/Converged_Param_Dictionary.json", 'r') as file:
    content = file.read().strip()  # Read and remove surrounding whitespace
    
    # Check if content is wrapped in double quotes
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1]  # Remove the leading and trailing quotes
        content = content.replace('\\"', '"')  # Replace escaped quotes with normal quotes

    try:
        data = json.loads(content)  # Now load as JSON
        print("Loaded JSON data:", data)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

# Step 2: Extract the array using the key
key = "b,CumulativeXUVFlux,final"
if isinstance(data, dict):  # Check if the loaded content is a dictionary
    daCumulativeXUVFlux = data.get(key)
    print("Extracted data:", daCumulativeXUVFlux)

    if daCumulativeXUVFlux:
        # Step 3: Plot the data if it's successfully extracted
        plt.hist(daCumulativeXUVFlux)
        plt.xlabel('Cumulative XUV Flux [W/m2]')
        plt.ylabel('Number')
        plt.show()
else:
    print("Loaded data is not a dictionary.")
