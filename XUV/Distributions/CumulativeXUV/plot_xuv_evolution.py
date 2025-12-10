#!/usr/bin/env python3
"""
Plot XUV luminosity evolution for 100 random samples from Engle and Ribas models.

This script randomly selects 100 vplanet outputs from each of the EngleBarnes
and RibasBarnes directories and creates a two-panel plot comparing XUV luminosity
evolution over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import random
from pathlib import Path
import vplot

def load_xuv_data(filepath):
    """
    Load XUV luminosity data from gj1132.star.forward file.

    Parameters
    ----------
    filepath : str
        Path to gj1132.star.forward file

    Returns
    -------
    time : np.ndarray
        Time array in years
    xuv_tot : np.ndarray
        Total XUV luminosity in LSUN (column 5)
    """
    try:
        data = np.loadtxt(filepath)
        time = data[:, 0]  # Column 1: Time
        xuv_tot = data[:, 4]  # Column 5: LXUVTot (0-indexed as column 4)
        return time, xuv_tot
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def get_output_subdirectories(output_dir):
    """
    Get all subdirectories in the output directory.

    Parameters
    ----------
    output_dir : str
        Path to output directory

    Returns
    -------
    subdirs : list
        List of subdirectory paths
    """
    subdirs = [d for d in glob.glob(os.path.join(output_dir, '*'))
               if os.path.isdir(d)]
    return subdirs


def main():
    """Main function to create XUV evolution comparison plot."""

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Get the directory of this script
    script_dir = Path(__file__).parent

    # Define output directories
    engle_output_dir = script_dir / "EngleBarnes" / "output"
    ribas_output_dir = script_dir / "RibasBarnes" / "output"

    # Get all subdirectories
    print("Getting subdirectories...")
    engle_subdirs = get_output_subdirectories(str(engle_output_dir))
    ribas_subdirs = get_output_subdirectories(str(ribas_output_dir))

    print(f"Found {len(engle_subdirs)} EngleBarnes directories")
    print(f"Found {len(ribas_subdirs)} RibasBarnes directories")

    # Randomly sample 100 from each
    engle_sample = random.sample(engle_subdirs, min(100, len(engle_subdirs)))
    ribas_sample = random.sample(ribas_subdirs, min(100, len(ribas_subdirs)))

    print(f"Sampling {len(engle_sample)} EngleBarnes and {len(ribas_sample)} RibasBarnes runs")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Plot EngleBarnes data
    print("\nPlotting EngleBarnes data...")
    engle_count = 0
    for subdir in engle_sample:
        filepath = os.path.join(subdir, "gj1132.star.forward")
        if os.path.exists(filepath):
            time, xuv_tot = load_xuv_data(filepath)
            if time is not None and xuv_tot is not None:
                ax1.plot(time, xuv_tot, color='k', alpha=0.15, linewidth=0.8)
                engle_count += 1

    # Plot RibasBarnes data
    print("Plotting RibasBarnes data...")
    ribas_count = 0
    for subdir in ribas_sample:
        filepath = os.path.join(subdir, "gj1132.star.forward")
        if os.path.exists(filepath):
            time, xuv_tot = load_xuv_data(filepath)
            if time is not None and xuv_tot is not None:
                ax2.plot(time, xuv_tot, color='k', alpha=0.15, linewidth=0.8)
                ribas_count += 1

    labelfont=20
    tickfont=14
    # Format left panel (EngleBarnes)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Time [years]', fontsize=labelfont)
    ax1.set_ylabel(r'Total XUV Luminosity [$L_\odot$]', fontsize=labelfont)
    ax1.set_title('Engle (2024)', fontsize=labelfont)
    ax1.tick_params(axis='both',labelsize=tickfont)
    #ax1.grid(True, alpha=0.3)
    #ax1.text(0.98, 0.02, f'N = {engle_count}', transform=ax1.transAxes,
    #         fontsize=10, ha='right', va='bottom',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Format right panel (RibasBarnes)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Time [years]', fontsize=labelfont)
    #ax2.set_ylabel(r'Total XUV Luminosity [$L_\odot$]', fontsize=labelfont)
    ax2.set_title('Ribas et al. (2005)', fontsize=labelfont)
    ax2.tick_params(axis='both',labelsize=tickfont)
    #ax2.grid(True, alpha=0.3)
    #ax2.text(0.98, 0.02, f'N = {ribas_count}', transform=ax2.transAxes,
    #         fontsize=10, ha='right', va='bottom',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Set matching y-axis limits for easy comparison
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Save figure
    output_path = script_dir / "XUVEvol.pdf"
    print(f"\nSaving figure to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print("Done! Figure saved.")
    #plt.show()


if __name__ == "__main__":
    main()
