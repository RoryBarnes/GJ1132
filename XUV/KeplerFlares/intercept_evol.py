#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def plot_simple_intercept(b1, b2, b3, filename='simple_intercept_plot.png'):
    """
    Simple function to plot y-intercept evolution.
    
    Parameters:
    -----------
    b1, b2, b3 : float
        Your fitted parameters
    filename : str
        Output filename
    """
    
    print(f"Plotting intercept evolution for parameters:")
    print(f"b1 = {b1:.4f} (age dependence)")
    print(f"b2 = {b2:.4f} (mass dependence)")  
    print(f"b3 = {b3:.4f} (baseline)")
    print(f"Model: intercept = {b1:.3f}×log(age) + {b2:.3f}×mass + {b3:.3f}")
    
    # Age range
    ages_myr = np.logspace(1, 4, 100)  # 0.01 to 10 Gyr
    log_ages = np.log10(ages_myr)
    
    # Different masses
    masses = [0.3, 0.5, 0.7, 1.0]
    colors = ['blue', 'green', 'orange', 'red']
    
    # Create plot
    plt.figure(figsize=(10, 7))
    
    for mass, color in zip(masses, colors):
        # Calculate intercept: b1*log_age + b2*mass + b3
        intercepts = b1 * log_ages + b2 * mass + b3
        #print(intercepts)
        
        plt.semilogx(ages_myr, intercepts, color=color, linewidth=2.5,
                    label=f'{mass:.1f} M☉')
    
    # Formatting
    plt.xlabel('Stellar Age [Myr]', fontsize=14)
    plt.ylabel('Y-Intercept (Normalization)', fontsize=14)
    plt.title(f'Flare Frequency Normalization vs Age\n' +
              f'b₁={b1:.3f}, b₂={b2:.3f}, b₃={b3:.3f}', fontsize=14)
    
    plt.xlim(10, 10000)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Stellar Mass', fontsize=12)
    plt.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()
    
    print(f"Plot saved as {filename}")
    
    # Print some example values
    print(f"\nExample intercept values:")
    print(f"Age [Gyr]    0.3 M☉     0.5 M☉     0.7 M☉     1.0 M☉")
    print("-" * 55)
    
    for age in [0.01, 0.1, 1.0, 5.0, 10.0]:
        log_age = np.log10(age)
        values = []
        for mass in masses:
            intercept = b1 * log_age + b2 * mass + b3
            values.append(f"{intercept:8.3f}")
        print(f"{age:8.2f}   {values[0]} {values[1]} {values[2]} {values[3]}")

if __name__ == "__main__":
    
    # ================================================================
    # CHANGE THESE PARAMETERS TO YOUR FITTED VALUES
    # ================================================================
    
    # New 6-parameter fit
    #b1_fitted = 4.69    # Age dependence of normalization
    #b2_fitted = -16.45    # Mass dependence of normalization  
    #b3_fitted = 19.45    # Baseline normalization

    # Jim's old fit
    #b1_fitted = 1.91734981    # Age dependence of normalization
    #b2_fitted = -24.57936264    # Mass dependence of normalization  
    #b3_fitted = 33.65312658    # Baseline normalization

    # New 3-parameter fit
    b1_fitted = -0.51    # Age dependence of normalization
    b2_fitted = 2.67    # Mass dependence of normalization  
    b3_fitted = 31.56    # Baseline normalization

    # ================================================================
    # CREATE THE PLOT
    # ================================================================
    
    print("Y-Intercept Evolution Plot")
    print("=" * 30)
    print("Model: log10(rate) = -1 * log_energy + intercept")
    print("       intercept = b1*log_age + b2*mass + b3")
    print()
    
    plot_simple_intercept(b1_fitted, b2_fitted, b3_fitted)
    
    print(f"\nPhysical interpretation:")
    if b1_fitted > 0:
        print(f"- b1 = {b1_fitted:.3f} > 0: Flare normalization INCREASES with age")
    else:
        print(f"- b1 = {b1_fitted:.3f} < 0: Flare normalization DECREASES with age")
    
    if b2_fitted > 0:
        print(f"- b2 = {b2_fitted:.3f} > 0: Flare normalization INCREASES with mass")
    else:
        print(f"- b2 = {b2_fitted:.3f} < 0: Flare normalization DECREASES with mass")
    
    print(f"- b3 = {b3_fitted:.3f}: Baseline normalization level")
    
    print(f"\nTo use your own parameters:")
    print(f"1. Edit the values of b1_fitted, b2_fitted, b3_fitted above")
    print(f"2. Run this script again")
    print(f"   OR")
    print(f"1. Run plot_intercept_evolution.py (auto-loads MCMC results)")