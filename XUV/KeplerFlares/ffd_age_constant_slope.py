#!/usr/bin/env python3
"""
Simplified script to recreate Figure 10 with your own fitted parameters.

Just replace the parameters below with your MCMC results and run!
"""

import numpy as np
import matplotlib.pyplot as plt
import vplot



def old_constants():
    a1 = -0.06596571
    a2 =  0.77855978
    a3 = -1.0475149
    b1 = 1.91734981
    b2 = -24.57936264
    b3 = 33.65312658

    return a1,a2,a3,b1,b2,b3

def PlotFFD(log_age,params,log_energies,ages_myr,mass,description,linestyle):
    colors = [vplot.colors.pale_blue,vplot.colors.purple,vplot.colors.orange,vplot.colors.red,vplot.colors.dark_blue,'k']

    if (description == 'old'):
        a1, a2, a3, b1, b2, b3 = params
    else:
        b1,b2,b3 = params

    for iAge in range(len(log_age)):
        if (description == 'old'):
            slope = a1 * log_age[iAge] + a2 * mass + a3
        else:
            slope = -1
        intercept = b1 * log_age[iAge] + b2 * mass + b3

        ffd = 10**(slope*log_energies + intercept)
        energies = 10**log_energies
        label = repr(ages_myr[iAge])+' Myr ('+description+')'
        plt.plot(energies,ffd,color=colors[iAge],linestyle=linestyle,label = label)

def Plot(params, mass=0.5, filename='ffd_comp_constant_slope.png'):
    """
    Simple function to recreate Figure 10 with your fitted parameters.
    
    Parameters:
    -----------
    params : list or array
        Your fitted parameters [a1, a2, a3, b1, b2, b3]
    mass : float
        Stellar mass in solar masses (default: 0.5)
    filename : str
        Output filename
    """
    
    # Energy range
    log_energies = np.linspace(33, 36, 100)
    
    # Ages to show
    ages_myr = [10,100,1000,10000]
    log_age = np.log10(ages_myr)

    # Create plot
    fig = plt.figure(figsize=(6.5, 6))

    old_params = old_constants()

    PlotFFD(log_age,old_params,log_energies,ages_myr,mass,'old','dashed')
    PlotFFD(log_age,params,log_energies,ages_myr,mass,'new','solid')

    # Formatting
    plt.xlabel('log Flare Energy (erg)', fontsize=18)
    plt.ylabel('Cumulative Flare Freq (#/day)', fontsize=18)
    
    plt.xlim(8e32, 1.05e36)
    plt.ylim(3e-6, 3e-1)
    #plt.ylim(1e-3, 1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([1e33,1e34,1e35,1e36],['33','34','35','36'],fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower left',fontsize=14)
    
    plt.annotate(f'M = 0.5 M$_\odot$', [2e33,8e-4],fontsize=14)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()
    
    print(f"Figure saved as {filename}")

if __name__ == "__main__":
    
    # Try to load parameters from MCMC results first
    try:
        param_samples = np.load('flare_mcmc_samples_fixed_slope.npy')
        
        # Calculate median parameters
        fitted_params = [
            np.median(param_samples[:, 0]),  # a1: Age dependence of energy slope  
            np.median(param_samples[:, 1]),  # a2: Mass dependence of energy slope
            np.median(param_samples[:, 2]),  # a3: Baseline energy slope
        ]
        
        # Also calculate uncertainties (standard deviation of posterior)
        uncertainties = [
            np.std(param_samples[:, 0]),
            np.std(param_samples[:, 1]), 
            np.std(param_samples[:, 2]),
        ]
        
        print("Parameter values (median ± std):")
        param_names = ['b1', 'b2', 'b3']
        for name, value, error in zip(param_names, fitted_params, uncertainties):
            print(f"  {name}: {value:.4f} ± {error:.4f}")
        
    except (FileNotFoundError, ImportError):
        print("Could not load MCMC results!")
        exit()

    Plot(fitted_params, mass=0.5)