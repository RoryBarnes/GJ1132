"""
Compare posteriors from dynesty and emcee samplers.

Creates a corner plot with contours from both samplers overlaid.
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import vplot

# Load samples from both samplers
dynesty_data = np.load("gj1132_dynesty/dynesty_samples_final_custom_iter_500.npz")
emcee_data = np.load("gj1132_emcee/emcee_samples_final_custom_iter_401.npz")

dynesty_samples = dynesty_data['samples']
emcee_samples = emcee_data['samples']

# Parameter labels
labels = [r"$m_{\star}$ [M$_{\odot}$]",
          r"$f_{sat}$",
          r"$t_{sat}$ [Gyr]",
          r"Age [Gyr]",
          r"$\beta_{XUV}$"]

# Parameter bounds for plot ranges
bounds = [(0.17, 0.22),         # dMass [Msun]
          (-4.0, -2.15),        # dSatXUVFrac (log10)
          (0.1, 5.0),           # dSatXUVTime [Gyr]
          (1.0, 13.0),          # dStopTime (age) [Gyr]
          (0.4, 2.1)]           # dXUVBeta

# Prior data - same as in gj1132_alabi.py
# Format: (mean, std) for symmetric Gaussian, (mean, std_pos, std_neg) for asymmetric
prior_data = [(0.1945, 0.0048, 0.0046),  # mass [Msun] - asymmetric Gaussian
              (-2.92, 0.26),              # log(fsat) - symmetric Gaussian
              (None, None),               # tsat [Gyr] - no prior (uniform)
              (5.75, 1.38),               # age [Gyr] - symmetric Gaussian
              (1.18, 0.31)]               # beta - symmetric Gaussian

print("="*70)
print("COMPARING DYNESTY AND EMCEE POSTERIORS")
print("="*70)
print(f"\nDynesty samples: {dynesty_samples.shape}")
print(f"Emcee samples:   {emcee_samples.shape}")

# Get vplot colors
# Use two distinct colors from the vplot palette
color_dynesty = vplot.colors.sPaleBlue
color_emcee = vplot.colors.sOrange

print(f"\nPlot colors:")
print(f"  Dynesty: pale blue")
print(f"  Emcee:   orange")

# Spacing parameter for vertical separation between panels
# Adjust this value to control panel spacing (smaller = less space)
dVerticalSpacing = 0.0  # Default is typically around 0.05

# Tick label fontsize - adjust to taste
tick_fontsize = 14

# Maximum likelihood parameter values from XUV/EvolutionPlots/MaximumLikelihood
# Order: [dMass, dSatXUVFrac, dSatXUVTime, dStopTime (age), dXUVBeta]
ml_params = np.array([0.194111, -2.920301, 0.423162, 5.752484, 1.180995])

# Create a square figure to ensure square panels
# The corner package will create ndim x ndim panels, so figure should accommodate that plus margins
figsize = (12, 12)

# Create figure with corner plot
fig = corner.corner(
    dynesty_samples,
    labels=labels,
    range=bounds,
    color=color_dynesty,
    bins=30,
    smooth=1.0,
    plot_datapoints=False,  # Don't show individual points
    plot_density=False,     # Don't show density
    fill_contours=False,    # Don't fill contours
    hist_kwargs={
        "density": True,
        "histtype": "step",
        "linewidth": 2.0,
        "color": color_dynesty
    },
    contour_kwargs={
        "linewidths": 2.0,
        "colors": color_dynesty
    },
    label_kwargs={
        "fontsize": 20
    },
    title_kwargs={
        "fontsize": 12
    },
    fig=plt.figure(figsize=figsize)
)

# Overlay emcee samples
corner.corner(
    emcee_samples,
    fig=fig,
    range=bounds,
    color=color_emcee,
    bins=30,
    smooth=1.0,
    plot_datapoints=False,
    plot_density=False,
    fill_contours=False,
    hist_kwargs={
        "density": True,
        "histtype": "step",
        "linewidth": 2.0,
        "color": color_emcee
    },
    contour_kwargs={
        "linewidths": 2.0,
        "colors": color_emcee
    }
)

# Add maximum likelihood points to all panels
# Get the axes from the figure
axes = np.array(fig.axes).reshape((5, 5))

# Set tick label fontsize for all panels
for i in range(5):
    for j in range(5):
        ax = axes[i, j]
        if i >= j:  # Only active panels (lower triangle + diagonal)
            ax.tick_params(axis='both', labelsize=tick_fontsize)

# Plot ML values on 1D and 2D panels
for i in range(5):
    # Diagonal panels (1D histograms)
    ax = axes[i, i]
    ymin, ymax = ax.get_ylim()
    ax.plot([ml_params[i], ml_params[i]], [0, ymax],
            color='k', linewidth=2.0, linestyle='-', alpha=0.8, zorder=10)

    # Off-diagonal panels (2D contours)
    for j in range(i):
        ax = axes[i, j]
        ax.plot(ml_params[j], ml_params[i], 'ko',
                markersize=8, markerfacecolor='k',
                markeredgewidth=1.5, markeredgecolor='k',
                alpha=0.8, zorder=10)

# Add prior distributions to the diagonal panels

# Plot priors on diagonal
color_prior = 'grey'

# Store posterior statistics for titles (to be added after spacing adjustment)
title_data = []

for i in range(5):
    ax = axes[i, i]
    xmin, xmax = bounds[i]
    x_range = np.linspace(xmin, xmax, 1000)

    if prior_data[i][0] is None:
        # Uniform prior - flat line
        y_uniform = np.ones_like(x_range) / (xmax - xmin)
        ax.plot(x_range, y_uniform, color=color_prior, linewidth=2.0,
                linestyle='--', alpha=0.7, zorder=0)
    elif len(prior_data[i]) == 2:
        # Symmetric Gaussian prior
        mean, std = prior_data[i]
        y_prior = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        ax.plot(x_range, y_prior, color=color_prior, linewidth=2.0,
                linestyle='--', alpha=0.7, zorder=0)
    elif len(prior_data[i]) == 3:
        # Asymmetric Gaussian prior
        mean, std_pos, std_neg = prior_data[i]
        y_prior = np.zeros_like(x_range)
        # Upper half (x >= mean)
        mask_upper = x_range >= mean
        y_prior[mask_upper] = (1.0 / (std_pos * np.sqrt(2 * np.pi))) * \
                              np.exp(-0.5 * ((x_range[mask_upper] - mean) / std_pos) ** 2)
        # Lower half (x < mean)
        mask_lower = x_range < mean
        y_prior[mask_lower] = (1.0 / (std_neg * np.sqrt(2 * np.pi))) * \
                              np.exp(-0.5 * ((x_range[mask_lower] - mean) / std_neg) ** 2)
        ax.plot(x_range, y_prior, color=color_prior, linewidth=2.0,
                linestyle='--', alpha=0.7, zorder=0)

    # Calculate posterior statistics for titles
    # Calculate median and asymmetric uncertainties (16th and 84th percentiles)
    dynesty_median = np.median(dynesty_samples[:, i])
    dynesty_q16, dynesty_q84 = np.percentile(dynesty_samples[:, i], [16, 84])
    dynesty_err_low = dynesty_median - dynesty_q16
    dynesty_err_high = dynesty_q84 - dynesty_median

    emcee_median = np.median(emcee_samples[:, i])
    emcee_q16, emcee_q84 = np.percentile(emcee_samples[:, i], [16, 84])
    emcee_err_low = emcee_median - emcee_q16
    emcee_err_high = emcee_q84 - emcee_median

    # Format title with colored text
    # Determine precision based on parameter range
    param_range = xmax - xmin
    if param_range < 0.1:
        fmt = '.4f'
    elif param_range < 1.0:
        fmt = '.3f'
    elif param_range < 10:
        fmt = '.2f'
    else:
        fmt = '.1f'

    # Create title strings
    dynesty_str = f"${dynesty_median:{fmt}}^{{+{dynesty_err_high:{fmt}}}}_{{-{dynesty_err_low:{fmt}}}}$"
    emcee_str = f"${emcee_median:{fmt}}^{{+{emcee_err_high:{fmt}}}}_{{-{emcee_err_low:{fmt}}}}$"

    # Store separate strings for later (will create colored text objects)
    title_data.append((i, dynesty_str, emcee_str))

# Add title
#fig.suptitle("Comparison of Dynesty and Emcee Posteriors",
#             fontsize=18, y=0.995)

# Adjust corner's default spacing - simpler than manual repositioning
# These values can be easily adjusted to control panel spacing and aspect ratio
fig.subplots_adjust(
    hspace=0.05,    # Vertical spacing between panels (adjust to taste)
    wspace=0.05,    # Horizontal spacing between panels
    left=0.12,      # Left margin for y-axis labels
    right=0.98,     # Right margin
    bottom=0.10,    # Bottom margin for x-axis labels
    top=0.95        # Top margin - leave room for titles
)

# Manually create legend using figure coordinates (completely independent of panel layout)
# This prevents matplotlib from adjusting panel spacing based on legend size
legend_fontsize = 20
legend_x_fig = 0.72  # Figure x-coordinate (0-1)
legend_y_fig = 0.87  # Figure y-coordinate (0-1)
legend_line_length = 0.03  # Line length in figure coordinates
legend_text_offset = 0.01  # Space between line and text
legend_y_spacing = 0.04  # Vertical spacing between legend entries

# Dynesty line and text
fig.lines.append(plt.Line2D([legend_x_fig, legend_x_fig + legend_line_length],
                            [legend_y_fig, legend_y_fig],
                            color=color_dynesty, linewidth=2, transform=fig.transFigure))
fig.text(legend_x_fig + legend_line_length + legend_text_offset, legend_y_fig, 'Dynesty',
         fontsize=legend_fontsize, va='center', transform=fig.transFigure)

# Emcee line and text
fig.lines.append(plt.Line2D([legend_x_fig, legend_x_fig + legend_line_length],
                            [legend_y_fig - legend_y_spacing, legend_y_fig - legend_y_spacing],
                            color=color_emcee, linewidth=2, transform=fig.transFigure))
fig.text(legend_x_fig + legend_line_length + legend_text_offset, legend_y_fig - legend_y_spacing, 'Emcee',
         fontsize=legend_fontsize, va='center', transform=fig.transFigure)

# Prior line and text
fig.lines.append(plt.Line2D([legend_x_fig, legend_x_fig + legend_line_length],
                            [legend_y_fig - 2*legend_y_spacing, legend_y_fig - 2*legend_y_spacing],
                            color=color_prior, linewidth=2, linestyle='--', alpha=0.7,
                            transform=fig.transFigure))
fig.text(legend_x_fig + legend_line_length + legend_text_offset, legend_y_fig - 2*legend_y_spacing,
         'Prior', fontsize=legend_fontsize, va='center', transform=fig.transFigure)

# Maximum likelihood point and text
fig.lines.append(plt.Line2D([legend_x_fig + legend_line_length/2],
                            [legend_y_fig - 3*legend_y_spacing],
                            marker='o', color='k', markersize=8, linewidth=0,
                            markerfacecolor='k', markeredgewidth=1.5, alpha=0.8,
                            transform=fig.transFigure))
fig.text(legend_x_fig + legend_line_length + legend_text_offset, legend_y_fig - 3*legend_y_spacing,
         'Maximum Likelihood', fontsize=legend_fontsize, va='center', transform=fig.transFigure)

# Add titles with colored text using ax.text()
title_fontsize = 14
title_y_position = 1.01      # Position above the axis
line_spacing = 0.2         # Spacing between title lines

# for i, dynesty_str, emcee_str in title_data:
#     ax = axes[i, i]

#     # Add dynesty text (top line) in pale blue
#     ax.text(0.5, title_y_position + line_spacing, dynesty_str,
#             transform=ax.transAxes,
#             fontsize=title_fontsize,
#             color=color_dynesty,
#             ha='center', va='bottom',
#             clip_on=False)

#     # Add emcee text (bottom line) in orange
#     ax.text(0.5, title_y_position, emcee_str,
#             transform=ax.transAxes,
#             fontsize=title_fontsize,
#             color=color_emcee,
#             ha='center', va='bottom',
#             clip_on=False)

# Save figure
output_file = "sampler_comparison.png"
fig.savefig(output_file, dpi=300)
print(f"\n✓ Saved comparison plot: {output_file}")

# Print summary statistics comparison
print("\n" + "="*70)
print("POSTERIOR COMPARISON")
print("="*70)

for i, label in enumerate(labels):
    dynesty_mean = np.mean(dynesty_samples[:, i])
    dynesty_std = np.std(dynesty_samples[:, i])
    emcee_mean = np.mean(emcee_samples[:, i])
    emcee_std = np.std(emcee_samples[:, i])

    # Check agreement (within ~1 sigma)
    diff = abs(dynesty_mean - emcee_mean)
    avg_std = (dynesty_std + emcee_std) / 2
    agreement = diff / avg_std

    print(f"\n{label}:")
    print(f"  Dynesty: {dynesty_mean:.6f} ± {dynesty_std:.6f}")
    print(f"  Emcee:   {emcee_mean:.6f} ± {emcee_std:.6f}")
    print(f"  Δ/σ:     {agreement:.2f}", end="")

    if agreement < 0.5:
        print(" (excellent agreement)")
    elif agreement < 1.0:
        print(" (good agreement)")
    elif agreement < 2.0:
        print(" (moderate agreement)")
    else:
        print(" (poor agreement - investigate!)")

print("\n" + "="*70)
print("Note: Δ/σ is the difference in means divided by average std dev")
print("Δ/σ < 1.0 indicates good sampler agreement")
print("="*70)
