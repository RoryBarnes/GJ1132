import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import vplot

fig = plt.figure(figsize=(3.25, 3))
cumulative_earth_flux = 9.759583e+15

# Step 1: Load and process JSON file
with open("output/Converged_Param_Dictionary.json", 'r') as file:
    content = file.read().strip()
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1]
        content = content.replace('\\"', '"')
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

# Step 2: Extract and process the array
key = "b,CumulativeXUVFlux,final"
if isinstance(data, dict):
    lower_bound = 20
    upper_bound = 3e3
    
    # Extract and clean data
    daCumulativeXUVFlux = data.get(key)
    daCumulativeXUVFlux_clean = pd.Series(daCumulativeXUVFlux).dropna()
    daCumulativeXUVFlux_filtered = np.array(daCumulativeXUVFlux_clean)
    daCumulativeXUVFlux_filtered = daCumulativeXUVFlux_filtered / cumulative_earth_flux
    
    # Filter data within bounds
    mask = (daCumulativeXUVFlux_filtered >= lower_bound) & (daCumulativeXUVFlux_filtered <= upper_bound)
    daCumulativeXUVFlux_filtered = daCumulativeXUVFlux_filtered[mask]
    
    # Transform data to log space
    log_data = np.log10(daCumulativeXUVFlux_filtered)
    
    # Create bins with equal widths in log space
    iNumBins = 50
    log_lower = np.log10(lower_bound)
    log_upper = np.log10(upper_bound)
    bin_width = (log_upper - log_lower) / iNumBins
    
    # Create bin edges in log space
    log_bin_edges = np.arange(log_lower, log_upper + bin_width, bin_width)
    
    # Transform bin edges back to linear space
    bin_edges = 10**log_bin_edges
    
    # Calculate histogram
    counts, _ = np.histogram(daCumulativeXUVFlux_filtered, bins=bin_edges)
    fractions = counts / len(daCumulativeXUVFlux_filtered)
    
    # Plot using the bin centers
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for log space
    plt.step(bin_centers, fractions, where='mid', color='k')
    
    # Set plot parameters
    plt.xlabel('Cumulative XUV Flux\nRelative to Modern Earth')
    plt.ylabel('Fraction')
    plt.xlim(lower_bound, upper_bound)
    plt.xscale('log')
    plt.ylim(0, 0.075)
    plt.savefig('GJ1132b_CumulativeXUVFlux.png', dpi=300)
else:
    print("Loaded data is not a dictionary.")