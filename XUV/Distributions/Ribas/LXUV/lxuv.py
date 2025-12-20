#!/usr/bin/env python3
"""
Calculate XUV luminosity distribution for GJ 1132 using updated X-ray observations.

This script performs Monte Carlo uncertainty propagation using:
- Observed X-ray luminosity: L_X = 9.96e25 ± 2.95e25 erg/s
- EUV-to-X-ray scaling relation with uncertainties
- 10,000 samples to build a converged probability distribution for L_XUV

Output includes:
- Statistics (mean, std, 95% CI) in LSUN units
- Distribution samples saved to file
- Two-panel plot: histogram with statistics + Q-Q plot for normality assessment
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import vplot

# Constants
L_SUN = 3.846e33  # Solar luminosity in erg/s

# Input parameters - observational constraints
L_X_MEAN = 9.96e25  # erg/s
L_X_STD = 2.95e25   # erg/s
SLOPE_MEAN = 0.821
SLOPE_STD = 0.041
INTERCEPT_MEAN = 28.16
INTERCEPT_STD = 0.05
C_OFFSET = 27.44

# Sampling parameters
N_SAMPLES = 10000

# Bolometric luminosity - asymmetric uncertainties
L_BOL_MEAN = 0.00477  # LSUN
L_BOL_SIGMA_PLUS = 0.00036  # LSUN (upper uncertainty)
L_BOL_SIGMA_MINUS = 0.00026  # LSUN (lower uncertainty)

# Figure configuration for normalized histograms
FIG_SIZE_X = 3.25
FIG_SIZE_Y = 3
N_BINS = 50  # Number of bins for histograms


def calculate_lxuv_distribution():
    """
    Calculate XUV luminosity distribution using Monte Carlo sampling.

    Process:
    1. Sample L_X from normal distribution
    2. Sample EUV scaling relation coefficients from normal distributions
    3. For each sample: calculate C, L_EUV, and L_XUV
    4. Filter out negative L_X values (unphysical)
    5. Return array of L_XUV values

    Returns
    -------
    lxuv_samples : np.ndarray
        Array of L_XUV values in erg/s (shape: <= N_SAMPLES)
    """
    # Generate random samples for all parameters
    lx_samples = np.random.normal(L_X_MEAN, L_X_STD, N_SAMPLES)
    slope_samples = np.random.normal(SLOPE_MEAN, SLOPE_STD, N_SAMPLES)
    intercept_samples = np.random.normal(INTERCEPT_MEAN, INTERCEPT_STD, N_SAMPLES)

    # Filter out negative L_X samples (unphysical)
    valid_mask = lx_samples > 0
    lx_samples = lx_samples[valid_mask]
    slope_samples = slope_samples[valid_mask]
    intercept_samples = intercept_samples[valid_mask]

    print(f"Samples after filtering (L_X > 0): {len(lx_samples):,} out of {N_SAMPLES:,} original samples")
    print(f"Percentage retained: {100 * len(lx_samples) / N_SAMPLES:.1f}%\n")

    # Calculate C = log10(L_X) - 27.44
    c_samples = np.log10(lx_samples) - C_OFFSET

    # Calculate log L_EUV = slope * C + intercept
    log_leuv_samples = slope_samples * c_samples + intercept_samples

    # Calculate L_EUV = 10^(log L_EUV)
    leuv_samples = 10.0 ** log_leuv_samples

    # Calculate total L_XUV = L_X + L_EUV
    lxuv_samples = lx_samples + leuv_samples

    return lxuv_samples


def compute_statistics(samples):
    """
    Compute and print statistics of the distribution.

    Parameters
    ----------
    samples : np.ndarray
        Array of sample values

    Returns
    -------
    mean : float
        Mean of the distribution
    std : float
        Standard deviation of the distribution
    ci : tuple
        95% confidence interval (lower, upper)
    """
    mean = np.mean(samples)
    std = np.std(samples)
    ci = np.percentile(samples, [2.5, 97.5])
    median = np.median(samples)

    # Additional statistics for normality assessment
    skewness = stats.skew(samples)
    kurtosis = stats.kurtosis(samples)

    # Shapiro-Wilk test for normality (use subset for efficiency)
    shapiro_stat, shapiro_p = stats.shapiro(samples[:5000])

    # Print results
    print("=" * 70)
    print("XUV Luminosity Distribution Statistics")
    print("=" * 70)
    print(f"\nMonte Carlo Results ({len(samples):,} samples):")
    print(f"Best fit (mean):           {mean:.4e} LSUN")
    print(f"Uncertainty (std):         {std:.4e} LSUN")
    print(f"95% Confidence Interval:   [{ci[0]:.4e}, {ci[1]:.4e}] LSUN")

    # Convert to erg/s for reference
    mean_ergs = mean * L_SUN
    std_ergs = std * L_SUN
    print(f"\nIn erg/s (for reference):")
    print(f"Mean:                      {mean_ergs:.4e} erg/s")
    print(f"Std Dev:                   {std_ergs:.4e} erg/s")

    print(f"\nAdditional Statistics:")
    print(f"Median:                    {median:.4e} LSUN")
    print(f"Skewness:                  {skewness:.4f}")
    print(f"Kurtosis:                  {kurtosis:.4f}")
    print(f"\nNormality Assessment:")
    print(f"Shapiro-Wilk p-value:      {shapiro_p:.6f}")
    print(f"Is L_XUV normally distributed? {'Yes (p > 0.05)' if shapiro_p > 0.05 else 'No (p < 0.05)'}")
    print(f"\nValue Range:")
    print(f"Minimum:                   {np.min(samples):.4e} LSUN")
    print(f"Maximum:                   {np.max(samples):.4e} LSUN")
    print("=" * 70)

    return mean, std, ci


def convert_to_lsun(lxuv_ergs):
    """
    Convert L_XUV from erg/s to solar luminosity units.

    Parameters
    ----------
    lxuv_ergs : np.ndarray or float
        XUV luminosity in erg/s

    Returns
    -------
    lxuv_lsun : np.ndarray or float
        XUV luminosity in LSUN
    """
    return lxuv_ergs / L_SUN


def plot_distribution(samples_lsun, mean_lsun, std_lsun, ci_lsun):
    """
    Create two-panel plot: histogram with statistics + Q-Q plot for normality.

    Parameters
    ----------
    samples_lsun : np.ndarray
        L_XUV samples in LSUN units
    mean_lsun : float
        Mean of distribution in LSUN
    std_lsun : float
        Standard deviation in LSUN
    ci_lsun : tuple
        95% confidence interval (lower, upper) in LSUN
    """
    fig = plt.figure(figsize=(6.5, 6))

    # Subplot 1: Histogram with statistics
    plt.subplot(2, 1, 1)
    plt.hist(samples_lsun, bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Add mean line
    plt.axvline(mean_lsun, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_lsun:.4e}')

    # Add 95% CI lines
    plt.axvline(ci_lsun[0], color='orange', linestyle=':', linewidth=2,
                label=f'95% CI: [{ci_lsun[0]:.4e}, {ci_lsun[1]:.4e}]')
    plt.axvline(ci_lsun[1], color='orange', linestyle=':', linewidth=2)

    plt.xlabel('$L_{XUV}$ [$L_\\odot$]', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('GJ 1132 XUV Luminosity Distribution', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)

    # Subplot 2: Q-Q plot for normality assessment
    plt.subplot(2, 1, 2)
    stats.probplot(samples_lsun, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Checking if $L_{XUV}$ follows Normal Distribution',
              fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save as PDF
    plt.savefig('lxuv_distribution.pdf', dpi=300, bbox_inches='tight')
    print("\nPlot saved to: lxuv_distribution.pdf")

    # Uncomment below to display plot
    # plt.show()


def save_samples(samples_lsun, filename):
    """
    Save L_XUV samples to a text file.

    Parameters
    ----------
    samples_lsun : np.ndarray
        L_XUV samples in LSUN units
    filename : str
        Output filename
    """
    np.savetxt(filename, samples_lsun)
    print(f"Samples saved to: {filename}")
    print(f"Number of samples: {len(samples_lsun):,}")


def sample_asymmetric_normal(mean, sigma_plus, sigma_minus, n_samples):
    """
    Sample from normal distribution with asymmetric uncertainties.

    Uses split-normal approach:
    - For samples above mean: use sigma_plus
    - For samples below mean: use sigma_minus

    Parameters
    ----------
    mean : float
        Mean of the distribution
    sigma_plus : float
        Standard deviation for values above mean (upper uncertainty)
    sigma_minus : float
        Standard deviation for values below mean (lower uncertainty)
    n_samples : int
        Number of samples to generate

    Returns
    -------
    samples : np.ndarray
        Array of samples from asymmetric normal distribution
    """
    # Generate uniform random values [0, 1]
    u = np.random.uniform(0, 1, n_samples)

    # Allocate output array
    samples = np.zeros(n_samples)

    # For each sample, randomly assign to upper or lower half with 50% probability each
    upper_mask = u > 0.5

    # Sample upper half: mean + |N(0, sigma_plus)|
    n_upper = np.sum(upper_mask)
    samples[upper_mask] = mean + np.abs(np.random.normal(0, sigma_plus, n_upper))

    # Sample lower half: mean - |N(0, sigma_minus)|
    n_lower = n_samples - n_upper
    samples[~upper_mask] = mean - np.abs(np.random.normal(0, sigma_minus, n_lower))

    return samples


def calculate_lxuv_lbol_distribution(lxuv_samples_lsun):
    """
    Calculate L_XUV/L_bol distribution.

    Combines L_XUV samples with L_bol samples drawn from asymmetric distribution.

    Parameters
    ----------
    lxuv_samples_lsun : np.ndarray
        L_XUV samples in LSUN units

    Returns
    -------
    ratio_samples : np.ndarray
        L_XUV/L_bol ratio samples (dimensionless)
    """
    n_samples = len(lxuv_samples_lsun)

    # Sample L_bol with asymmetric uncertainties
    lbol_samples = sample_asymmetric_normal(
        L_BOL_MEAN, L_BOL_SIGMA_PLUS, L_BOL_SIGMA_MINUS, n_samples
    )

    # Calculate ratio
    ratio_samples = lxuv_samples_lsun / lbol_samples

    return ratio_samples


def plot_normalized_histogram(samples, mean, std, xlabel, filename, x_scale=1.0):
    """
    Create normalized histogram with Gaussian overlay.

    Follows the NormalizedHistogram pattern from age.py.

    Parameters
    ----------
    samples : np.ndarray
        Sample values
    mean : float
        Mean of distribution
    std : float
        Standard deviation of distribution
    xlabel : str
        X-axis label
    filename : str
        Output filename (PDF)
    x_scale : float, optional
        Scale factor for x-axis (e.g., 1e7 to display in units of 1e-7)
        Default is 1.0 (no scaling)
    """
    fig = plt.figure(figsize=(FIG_SIZE_X, FIG_SIZE_Y))

    # Scale the data
    scaled_samples = samples * x_scale
    scaled_mean = mean * x_scale
    scaled_std = std * x_scale

    # Create histogram
    counts, bin_edges = np.histogram(scaled_samples, bins=N_BINS)
    normalized_fractions = counts / len(scaled_samples)

    # Plot histogram as step function
    plt.step(bin_edges[:-1], normalized_fractions, where='mid',
             color='k', linewidth=1.5, label='Data')

    # Overlay Gaussian
    x_gauss = np.linspace(bin_edges[0], bin_edges[-1], 200)
    # Convert Gaussian PDF to match normalized fractions
    # PDF needs to be scaled by bin width
    bin_width = bin_edges[1] - bin_edges[0]
    gauss_pdf = stats.norm.pdf(x_gauss, scaled_mean, scaled_std) * bin_width
    plt.plot(x_gauss, gauss_pdf, color=vplot.colors.red, linestyle='dashed',linewidth=1.5,
             #label=f'Gaussian ($\\mu$={mean:.2e}, $\\sigma$={std:.2e})')
             label="Fit")

    # Formatting
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Fraction', fontsize=14)
    #plt.title(title, fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    # Save
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to: {filename}")

    # Close to free memory
    plt.close()


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("GJ 1132 XUV Luminosity Distribution Calculator")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  X-ray luminosity:      {L_X_MEAN:.3e} ± {L_X_STD:.3e} erg/s")
    print(f"  EUV slope:             {SLOPE_MEAN:.3f} ± {SLOPE_STD:.3f}")
    print(f"  EUV intercept:         {INTERCEPT_MEAN:.2f} ± {INTERCEPT_STD:.2f}")
    print(f"  Number of samples:     {N_SAMPLES:,}")
    print("=" * 70 + "\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Calculate distribution in erg/s
    print("Calculating L_XUV distribution...")
    lxuv_samples_ergs = calculate_lxuv_distribution()

    # Convert to LSUN for VPLanet compatibility
    lxuv_samples_lsun = convert_to_lsun(lxuv_samples_ergs)

    # Compute and print statistics
    mean_lsun, std_lsun, ci_lsun = compute_statistics(lxuv_samples_lsun)

    # Create visualization
    print("\nCreating plots...")
    plot_distribution(lxuv_samples_lsun, mean_lsun, std_lsun, ci_lsun)

    # Save samples to file
    print("Saving sample data...")
    save_samples(lxuv_samples_lsun, 'lxuv_samples.txt')

    # ===== NEW: Step 1 - Normalized histogram for L_XUV =====
    print("\nCreating normalized histogram for L_XUV...")
    plot_normalized_histogram(
        lxuv_samples_lsun, mean_lsun, std_lsun,
        xlabel='$L_{XUV}$ [$10^{-7} L_\\odot$]',
        filename='lxuv_hist.pdf',
        x_scale=1e7
    )

    # ===== NEW: Step 2 - Calculate L_XUV/L_bol distribution =====
    print("\nCalculating L_XUV/L_bol distribution...")
    ratio_samples = calculate_lxuv_lbol_distribution(lxuv_samples_lsun)

    # Compute ratio statistics
    ratio_mean = np.mean(ratio_samples)
    ratio_std = np.std(ratio_samples)
    ratio_ci = np.percentile(ratio_samples, [2.5, 97.5])

    print(f"L_XUV/L_bol Statistics:")
    print(f"  Mean:  {ratio_mean:.4e}")
    print(f"  Std:   {ratio_std:.4e}")
    print(f"  95% CI: [{ratio_ci[0]:.4e}, {ratio_ci[1]:.4e}]")

    # Save ratio samples
    print("\nSaving L_XUV/L_bol sample data...")
    save_samples(ratio_samples, 'lxuv_lbol_samples.txt')

    # ===== NEW: Step 3 - Normalized histogram for L_XUV/L_bol =====
    print("\nCreating normalized histogram for L_XUV/L_bol...")
    plot_normalized_histogram(
        ratio_samples, ratio_mean, ratio_std,
        xlabel='$L_{XUV} / L_{bol}$',
        #title='GJ 1132 XUV to Bolometric Luminosity Ratio',
        filename='lxuv_lbol_hist.pdf'
    )

    # ===== NEW: Step 4 - Log10 of L_XUV/L_bol distribution =====
    print("\nCalculating log10(L_XUV/L_bol) distribution...")
    log_ratio_samples = np.log10(ratio_samples)

    # Compute log ratio statistics
    log_ratio_mean = np.mean(log_ratio_samples)
    log_ratio_std = np.std(log_ratio_samples)
    log_ratio_ci = np.percentile(log_ratio_samples, [2.5, 97.5])

    print(f"log10(L_XUV/L_bol) Statistics:")
    print(f"  Mean:  {log_ratio_mean:.4f}")
    print(f"  Std:   {log_ratio_std:.4f}")
    print(f"  95% CI: [{log_ratio_ci[0]:.4f}, {log_ratio_ci[1]:.4f}]")

    # Save log ratio samples
    print("\nSaving log10(L_XUV/L_bol) sample data...")
    save_samples(log_ratio_samples, 'log_lxuv_lbol_samples.txt')

    # ===== NEW: Step 5 - Normalized histogram for log10(L_XUV/L_bol) =====
    print("\nCreating normalized histogram for log10(L_XUV/L_bol)...")
    plot_normalized_histogram(
        log_ratio_samples, log_ratio_mean, log_ratio_std,
        xlabel='$\log_{10}(L_{XUV} / L_{bol})$',
        filename='log_lxuv_lbol_hist.pdf'
    )

    print("\nAnalysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
