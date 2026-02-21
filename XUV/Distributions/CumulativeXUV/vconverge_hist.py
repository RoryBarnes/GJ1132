import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import vplot
from scipy import stats

fig = plt.figure(figsize=(6.5, 6))
ymax=0.25
lower_bound = 20
upper_bound = 4e3
dConfidenceInterval = 95
dirs = ['Engle', 'EngleBarnes', 'Ribas', 'RibasBarnes']

dCumulativeEarthFlux = 9.759583e+15
dShorelineFlux = 51.43

def GatherFluxes(file):
    #print(file,flush=True)

    file = file+'/output/Converged_Param_Dictionary.json'
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
    key = "b,CumulativeXUVFlux,final"
    if isinstance(data, dict):
    
        # Extract and clean data
        daCumulativeXUVFlux = data.get(key)
        daCumulativeXUVFlux_clean = pd.Series(daCumulativeXUVFlux).dropna()
        daCumulativeXUVFlux_filtered = np.array(daCumulativeXUVFlux_clean)
        daCumulativeXUVFlux_filtered = daCumulativeXUVFlux_filtered / dCumulativeEarthFlux
        
        # Filter data within bounds
        mask = (daCumulativeXUVFlux_filtered >= lower_bound) & (daCumulativeXUVFlux_filtered <= upper_bound)
        daCumulativeXUVFlux_filtered = daCumulativeXUVFlux_filtered[mask]
        
        # Calculate statistics on the original (linear) filtered data
        dMean = np.mean(daCumulativeXUVFlux_filtered)
        
        # Calculate 95% confidence interval for the mean
        dMin = (100-dConfidenceInterval)/2
        dMax = dConfidenceInterval + dMin
        dLower = np.percentile(daCumulativeXUVFlux_filtered, dMin)
        dUpper = np.percentile(daCumulativeXUVFlux_filtered, dMax)

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
        dFractions = counts / len(daCumulativeXUVFlux_filtered)
        
        # Plot using the bin centers
        dBinCenters = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for log space

        return dBinCenters, dFractions, dMean, dLower, dUpper

    else:
        print("Loaded data is not a dictionary.")

daEngleBins, daEngleFractions, dMeanEngle, dLowerEngle, dUpperEngle = GatherFluxes(dirs[0])
daEngleBarnesBins, daEngleBarnesFractions, dMeanEngleBarnes, dLowerEngleBarnes, dUpperEngleBarnes = GatherFluxes(dirs[1])
daRibasBins, daRibasFractions, dMeanRibas, dLowerRibas, dUpperRibas = GatherFluxes(dirs[2])
daRibasBarnesBins, daRibasBarnesFractions, dMeanRibasBarnes, dLowerRibasBarnes, dUpperRibasBarnes = GatherFluxes(dirs[3])

plt.axvline(dShorelineFlux, color=vplot.colors.pale_blue, linewidth=6)

plt.step(daEngleBins, daEngleFractions, where='mid', color='grey', linestyle='-', linewidth=2, label="Engle Only")
plt.step(daEngleBarnesBins, daEngleBarnesFractions, where='mid', color='k', linestyle='-', linewidth=2, label="Engle w/Flares")
plt.step(daRibasBins, daRibasFractions, where='mid', color=vplot.colors.orange, alpha=0.5,linestyle='-', linewidth=2, label="Ribas Only")
plt.step(daRibasBarnesBins, daRibasBarnesFractions, where='mid', color=vplot.colors.orange, linestyle='-', linewidth=2, label="Ribas w/Flares")

"""
plt.axvline(dMeanEngle, color='grey', linestyle=':', linewidth=1.5)
plt.axvline(dLowerEngle, color='grey', linestyle=':', linewidth=1)
plt.axvline(dUpperEngle, color='grey', linestyle=':', linewidth=1)

plt.axvline(dMeanEngleDavenport, color='k', linestyle=':', linewidth=1.5)
plt.axvline(dLowerEngleDavenport, color='k', linestyle=':', linewidth=1)
plt.axvline(dUpperEngleBarnes, color='k', linestyle=':', linewidth=1)

plt.axvline(dMeanRibas, color=vplot.colors.pale_blue, linestyle=':', linewidth=1.5)
plt.axvline(dLowerRibas, color=vplot.colors.pale_blue, linestyle=':', linewidth=1)
plt.axvline(dUpperRibas, color=vplot.colors.pale_blue, linestyle=':', linewidth=1)
"""

plt.xlabel('Normalized Cumulative XUV Flux',fontsize=24)
plt.ylabel('Fraction',fontsize=24)
plt.xlim(lower_bound, upper_bound)
plt.xscale('log')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, ymax)
plt.legend(loc='upper left',fontsize=16,  framealpha=0.9)

plt.annotate('Cosmic Shoreline',(40,0.06),fontsize=20,rotation=90,color=vplot.colors.pale_blue)
plt.savefig('GJ1132b_CumulativeXUV_Multi.png', dpi=300)

print(f"Engle Only - Mean: {dMeanEngle:.2f}, 95% CI: [{dLowerEngle:.2f}, {dUpperEngle:.2f}]")
print(f"Engle w/Flares - Mean: {dMeanEngleBarnes:.2f}, 95% CI: [{dLowerEngleBarnes:.2f}, {dUpperEngleBarnes:.2f}]")
print(f"Ribas Only - Mean: {dMeanRibas:.2f}, 95% CI: [{dLowerRibas:.2f}, {dUpperRibas:.2f}]")
print(f"Ribas w/Flares - Mean: {dMeanRibasBarnes:.2f}, 95% CI: [{dLowerRibasBarnes:.2f}, {dUpperRibasBarnes:.2f}]")
