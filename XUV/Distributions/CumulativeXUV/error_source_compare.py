import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import vplot
from scipy import stats

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
ymax = 0.6
lower_bound = 20
upper_bound = 4e3
dConfidenceInterval = 95

dCumulativeEarthFlux = 9.759583e+15
dShorelineFlux = 51.43

# Define data directories for left panel (Engle variants)
engle_dirs = ['Engle', 'EngleModelErrorsOnly', 'EngleStellarErrorsOnly']
engle_labels = ['All Errors', 'Model Errors Only', 'Stellar Errors Only']
engle_colors = ['k', vplot.colors.purple, vplot.colors.red]

# Define data directories for right panel (Ribas variants)
ribas_dirs = ['Ribas', 'RibasModelErrorsOnly', 'RibasStellarErrorsOnly']
ribas_labels = ['All Errors', 'Model Errors Only', 'Stellar Errors Only']
ribas_colors = ['k', vplot.colors.purple, vplot.colors.red]


def GatherFluxes(file):
    """Load and process cumulative XUV flux data from JSON file."""
    file = file + '/output/Converged_Param_Dictionary.json'
    with open(file, 'r') as f:
        content = f.read().strip()
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
            content = content.replace('\\"', '"')
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)

    # Extract and process the array
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
        dMin = (100 - dConfidenceInterval) / 2
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


# Load Engle variants
engle_data = []
for directory in engle_dirs:
    bins, fractions, mean, lower, upper = GatherFluxes(directory)
    engle_data.append((bins, fractions, mean, lower, upper))

# Load Ribas variants
ribas_data = []
for directory in ribas_dirs:
    bins, fractions, mean, lower, upper = GatherFluxes(directory)
    ribas_data.append((bins, fractions, mean, lower, upper))

# Plot left panel (Engle variants)
ax1.axvline(dShorelineFlux, color=vplot.colors.pale_blue, linewidth=6)
for i, (bins, fractions, mean, lower, upper) in enumerate(engle_data):
    ax1.step(bins, fractions, where='mid', color=engle_colors[i], linestyle='-',
             linewidth=2, label=engle_labels[i])

ax1.set_xlabel('Normalized Cumulative XUV Flux', fontsize=20)
ax1.set_ylabel('Fraction', fontsize=20)
ax1.set_xlim(lower_bound, upper_bound)
ax1.set_xscale('log')
ax1.tick_params(axis='both', labelsize=16)
ax1.set_ylim(0, ymax)
ax1.legend(loc='upper right', fontsize=12)
ax1.set_title('Engle (2024)', fontsize=22)
ax1.annotate('Cosmic Shoreline', (40, 0.06), fontsize=20, rotation=90,
             color=vplot.colors.pale_blue)

# Plot right panel (Ribas variants)
ax2.axvline(dShorelineFlux, color=vplot.colors.pale_blue, linewidth=6)
for i, (bins, fractions, mean, lower, upper) in enumerate(ribas_data):
    ax2.step(bins, fractions, where='mid', color=ribas_colors[i], linestyle='-',
             linewidth=2, label=ribas_labels[i])

ax2.set_xlabel('Normalized Cumulative XUV Flux', fontsize=20)
ax2.set_ylabel('Fraction', fontsize=20)
ax2.set_xlim(lower_bound, upper_bound)
ax2.set_xscale('log')
ax2.tick_params(axis='both', labelsize=16)
ax2.set_ylim(0, ymax)
ax2.legend(loc='upper right', fontsize=12)
ax2.set_title('Ribas et al. (2005)', fontsize=22)
ax2.annotate('Cosmic Shoreline', (40, 0.06), fontsize=20, rotation=90,
             color=vplot.colors.pale_blue)

plt.tight_layout()
plt.savefig('GJ1132b_ErrorSourceComparison.png', dpi=300)

# Print statistics
print("ENGLE MODEL VARIANTS:")
for i, label in enumerate(engle_labels):
    mean, lower, upper = engle_data[i][2], engle_data[i][3], engle_data[i][4]
    print(f"  {label:25s} - Mean: {mean:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")

print("\nRIBAS MODEL VARIANTS:")
for i, label in enumerate(ribas_labels):
    mean, lower, upper = ribas_data[i][2], ribas_data[i][3], ribas_data[i][4]
    print(f"  {label:25s} - Mean: {mean:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")
