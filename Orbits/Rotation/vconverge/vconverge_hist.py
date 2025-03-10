import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import vplot

fig = plt.figure(figsize=(3.25, 3))
lower_bound = 1e-7
upper_bound = 1e-3


def GatherObliquities(file):
# Step 1: Load and process JSON file
    with open(file, 'r') as file:
        content = file.read().strip()
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
            content = content.replace('\\"', '"')
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)

    # Step 2: Extract and process the array
    key = "b,Obliquity,final"
    if isinstance(data, dict):
        # Extract and clean data
        daObliquity = data.get(key)
        daObliquity_clean = pd.Series(daObliquity).dropna()
        daObliquity_filtered = np.array(daObliquity_clean)
        
        # Filter data within bounds
        mask = (daObliquity_filtered >= lower_bound) & (daObliquity_filtered <= upper_bound)
        daObliquity_filtered = daObliquity_filtered[mask]

        # Transform data to log space
        log_data = np.log10(daObliquity_filtered)
        
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
        counts, _ = np.histogram(daObliquity_filtered, bins=bin_edges)
        fractions = counts / len(daObliquity_filtered)
    
        # Plot using the bin centers
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for log space

        return bin_centers, fractions
    else:
        print("Loaded data is not a dictionary.")

daCPLBins, daCPLFractions = GatherObliquities("CPL/output/Converged_Param_Dictionary.json")
daCTLBins, daCTLFractions = GatherObliquities("CTL/output/Converged_Param_Dictionary.json")

plt.step(daCPLBins,daCPLFractions, where='mid', color=vplot.colors.pale_blue,label='CPL')
plt.step(daCTLBins,daCTLFractions, where='mid', color=vplot.colors.red,label='CPL')

# Set plot parameters
plt.xlabel(r'Equilibrium Obliquity ($^\circ$)')
plt.ylabel('Fraction')
plt.xlim(lower_bound, upper_bound)
plt.xscale('log')
plt.xticks([1e-7,1e-6,1e-5,1e-4,1e-3], ['$10^{-7}$', '$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-3}$'])
plt.ylim(0, 0.1)
plt.legend(loc='best')

plt.savefig('GJ1132b_Obliquity_Hist.png', dpi=300)
