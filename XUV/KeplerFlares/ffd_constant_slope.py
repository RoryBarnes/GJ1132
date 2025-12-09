#!/usr/bin/env python3
"""
Bayesian inference of stellar flare frequency distribution parameters for ensemble data.
SIMPLIFIED MODEL with fixed slope of -1.

This script uses emcee to infer posterior distributions for parameters
b1, b2, b3 in the simplified flare frequency distribution model using
data from multiple stars in an ensemble.

SIMPLIFIED MODEL:
Model: log10(rate) = -1 * log_energy + (b1*log_age + b2*mass + b3)

Where:
- Slope is fixed at -1 (universal power-law)
- b1, b2, b3: Control the normalization (intercept dependence on age/mass)

This reduces the parameter space from 6 to 3 parameters compared to the full
Davenport et al. 2019 model.
"""

import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import corner

def log_flare_rate_model(log_energy, log_age, mass, params):
    """
    Simplified model for log10(flare rate) with fixed slope = -1.
    
    Model: log10(rate) = -1 * log_energy + (b1*log_age + b2*mass + b3)
    
    This assumes a universal power-law slope of -1 for the energy distribution,
    with only the normalization depending on stellar age and mass.
    
    Parameters:
    -----------
    log_energy : float or array
        Base-10 logarithm of flare energy (erg)
    log_age : float or array  
        Base-10 logarithm of stellar age (Gyr)
    mass : float or array
        Stellar mass (solar masses)
    params : array-like
        Model parameters [b1, b2, b3] (only intercept parameters)
        
    Returns:
    --------
    log_rate : float or array
        Base-10 logarithm of predicted flare rate
    """
    b1, b2, b3 = params
    
    # Fixed slope of -1
    slope = -1.0
    
    # Intercept term (depends on age and mass) - controls normalization
    intercept = b1 * log_age + b2 * mass + b3
    
    # Linear model: log(rate) = slope * log(energy) + intercept
    log_rate = slope * log_energy + intercept
    
    return log_rate

def log_likelihood_ensemble(params, log_energy, log_age, mass, observed_log_ff, log_ff_errors):
    """
    Log-likelihood function for the ensemble flare frequency model.
    
    This version handles data from multiple stars properly by computing
    the likelihood for the entire ensemble dataset.
    
    Parameters:
    -----------
    params : array-like
        Model parameters [a1, a2, a3, b1, b2, b3]
    log_energy : array
        Base-10 logarithm of flare energies
    log_age : array
        Base-10 logarithm of stellar ages
    mass : array
        Stellar masses
    observed_log_ff : array
        Observed log10(flare frequency)
    log_ff_errors : array
        Uncertainties in log10(flare frequency)
        
    Returns:
    --------
    log_like : float
        Log-likelihood value
    """
    # Predict flare rates for all data points
    predicted_log_ff = log_flare_rate_model(log_energy, log_age, mass, params)
    
    # Calculate residuals
    residuals = observed_log_ff - predicted_log_ff
    
    # Gaussian likelihood
    chi_squared = np.sum((residuals / log_ff_errors) ** 2)
    log_like = -0.5 * (chi_squared + np.sum(np.log(2 * np.pi * log_ff_errors ** 2)))
    
    # Check for numerical issues
    if not np.isfinite(log_like):
        return -np.inf
    
    return log_like

def log_prior_ensemble(params):
    """
    Log-prior function with physically motivated priors for the simplified model.
    
    Parameters:
    -----------
    params : array-like
        Model parameters [b1, b2, b3] (only intercept parameters)
        b1: Age dependence of intercept
        b2: Mass dependence of intercept  
        b3: Baseline intercept
        
    Returns:
    --------
    log_p : float
        Log-prior probability
    """
    b1, b2, b3 = params
    
    # Priors for the simplified model (slope fixed at -1)
    if (-20 < b1 < 20 and           # Age dependence of intercept
        -20 < b2 < 20 and           # Mass dependence of intercept  
        -30 < b3 < 40):             # Baseline intercept (broad range)
        return 0.0  # Uniform prior
    else:
        return -np.inf  # Outside prior bounds

def log_posterior_ensemble(params, log_energy, log_age, mass, observed_log_ff, log_ff_errors):
    """
    Log-posterior probability function for ensemble analysis.
    """
    lp = log_prior_ensemble(params)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood_ensemble(params, log_energy, log_age, mass, observed_log_ff, log_ff_errors)
    
    return lp + ll

def find_initial_guess_ensemble(log_energy, log_age, mass, observed_log_ff, log_ff_errors):
    """
    Find initial parameter guess using maximum likelihood estimation for the simplified model.
    
    Returns:
    --------
    initial_params : array
        Best-fit parameters to use as starting point for MCMC [b1, b2, b3]
    """
    
    def neg_log_likelihood(params):
        return -log_likelihood_ensemble(params, log_energy, log_age, mass, observed_log_ff, log_ff_errors)
    
    # Initial guess for simplified model (slope fixed at -1)
    # b1, b2, b3 control intercept
    p0 = [-2.0, 5.0, 15.0]  # Reasonable starting point
    
    # Bounds for optimization (matching the simplified priors)
    bounds = [(-20, 20), (-20, 20), (-30, 40)]
    
    print("Optimizing initial parameters for simplified model (slope = -1)...")
    try:
        # Use differential evolution for more robust global optimization
        result = differential_evolution(neg_log_likelihood, bounds, seed=42, maxiter=500)
        
        if result.success:
            print(f"Initial optimization successful. Final likelihood: {-result.fun:.2f}")
            return result.x
        else:
            print("Warning: Initial optimization did not converge fully.")
            return result.x
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        print("Using default initial guess")
        return np.array(p0)

def load_and_process_ensemble_data(data_file):
    """
    Load and process the ensemble flare frequency data.
    
    Parameters:
    -----------
    data_file : str
        Path to CSV file with ensemble flare data
        
    Returns:
    --------
    data_arrays : tuple
        (log_energy, log_age, mass, observed_log_ff, log_ff_errors, star_ids)
    """
    
    print("Loading ensemble data...")
    try:
        data = pd.read_csv(data_file)
        print(f"Loaded {len(data)} rows from {data_file}")
    except FileNotFoundError:
        print(f"Error: Could not find {data_file}. Using sample data for demonstration...")
        # Use sample data if file not found
        data_str = """logE,logAge,mass,giclr,Prot,FF,FFerr,logFF,logFFerr
35.34892020701608,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.0025314096480615386,0.0021921240538215455,-2.596637568987615,inf
35.148920207016076,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.007036270286523077,0.002536582324484261,-2.1526574861999483,inf
34.94892020701607,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.023907653795607695,0.004147031799937468,-1.6214630417686646,inf
34.74892020701608,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.06483541443878463,0.005795819712855208,-1.1881877087272184,inf
34.548920207016074,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.14304210179886157,0.007897060298653792,-0.8445361171657367,inf
34.34892020701608,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.25889511416534616,0.009571461421119146,-0.5868761454363197,inf
34.148920207016076,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.3392218030355154,0.008164820017038414,-0.46951624181124124,inf
33.94892020701607,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.35545834820083205,0.003077491859003192,-0.449211281590533,inf
34.5777686221268,-0.3214820754570555,0.803880852947368,1.0490000000000013,0.215,0.0021430028144,0.003738156698102526,-2.668977258599988,inf
34.377768622126794,-0.3214820754570555,0.803880852947368,1.0490000000000013,0.215,0.0043695185408999995,0.0032664607840441474,-2.3595664134986,inf
34.1777686221268,-0.3214820754570555,0.803880852947368,1.0490000000000013,0.215,0.0043695185408999995,0.0026240314624560173,-2.3595664134986,inf"""
        
        from io import StringIO
        data = pd.read_csv(StringIO(data_str))
        print(f"Using sample data with {len(data)} rows")
    
    print("Processing ensemble data...")
    
    # Filter out infinite, zero, and invalid flare frequency values
    valid_mask = (
        (data['FF'] > 0) & 
        np.isfinite(data['FF']) & 
        np.isfinite(data['FFerr']) & 
        (data['FFerr'] > 0) &
        np.isfinite(data['logAge']) &
        np.isfinite(data['mass']) &
        np.isfinite(data['logE'])
    )
    
    data_clean = data[valid_mask].copy()
    
    if len(data_clean) == 0:
        raise ValueError("No valid flare frequency data found!")
    
    # Create unique star identifiers based on stellar properties
    # Stars with same age, mass, and rotation period are considered the same
    stellar_props = ['logAge', 'mass', 'Prot']
    data_clean['star_id'] = data_clean.groupby(stellar_props).ngroup()
    
    # Get ensemble statistics
    n_stars = data_clean['star_id'].nunique()
    n_total_points = len(data_clean)
    
    print(f"Ensemble statistics:")
    print(f"  Total valid data points: {n_total_points}")
    print(f"  Number of stars: {n_stars}")
    print(f"  Average points per star: {n_total_points/n_stars:.1f}")
    
    # Extract variables for the full ensemble
    log_energy = data_clean['logE'].values
    log_age = data_clean['logAge'].values  
    mass = data_clean['mass'].values
    observed_ff = data_clean['FF'].values
    ff_errors = data_clean['FFerr'].values
    star_ids = data_clean['star_id'].values
    
    # Convert to log space for fitting
    observed_log_ff = np.log10(observed_ff)
    
    # Estimate errors in log space (using error propagation)
    log_ff_errors = ff_errors / (observed_ff * np.log(10))
    
    # Set minimum error floor to avoid numerical issues
    min_log_error = 0.01  # ~2.3% relative error
    log_ff_errors = np.maximum(log_ff_errors, min_log_error)
    
    print(f"Data ranges:")
    print(f"  Energy: {log_energy.min():.2f} to {log_energy.max():.2f} (log10 erg)")
    print(f"  Age: {log_age.min():.3f} to {log_age.max():.3f} (log10 Gyr)")
    print(f"  Mass: {mass.min():.2f} to {mass.max():.2f} (solar masses)")
    print(f"  Flare rate: {observed_log_ff.min():.2f} to {observed_log_ff.max():.2f} (log10 events/day)")
    
    # Check for potential issues
    age_range = log_age.max() - log_age.min()
    mass_range = mass.max() - mass.min()
    
    if age_range < 0.1:
        print(f"Warning: Limited age range ({age_range:.3f} dex). Age evolution may be poorly constrained.")
    if mass_range < 0.1:
        print(f"Warning: Limited mass range ({mass_range:.2f} M☉). Mass dependence may be poorly constrained.")
    
    return log_energy, log_age, mass, observed_log_ff, log_ff_errors, star_ids

def run_mcmc_ensemble(data_file, nwalkers=32, nsteps=5000, burn_in=1000, thin=10):
    """
    Run MCMC sampling to infer flare frequency parameters from ensemble data.
    
    Parameters:
    -----------
    data_file : str
        Path to CSV file with ensemble flare data
    nwalkers : int
        Number of MCMC walkers
    nsteps : int  
        Number of MCMC steps
    burn_in : int
        Number of burn-in steps to discard
    thin : int
        Thinning factor for chain
        
    Returns:
    --------
    sampler : emcee.EnsembleSampler
        MCMC sampler object with results
    """
    
    # Load and process ensemble data
    log_energy, log_age, mass, observed_log_ff, log_ff_errors, star_ids = load_and_process_ensemble_data(data_file)
    
    # Find initial parameter estimate
    print("Finding initial parameter guess...")
    initial_params = find_initial_guess_ensemble(log_energy, log_age, mass, observed_log_ff, log_ff_errors)
    print(f"Initial parameters (simplified model): b1={initial_params[0]:.3f}, b2={initial_params[1]:.3f}, b3={initial_params[2]:.3f}")
    
    # Set up MCMC
    ndim = 3  # Number of parameters (reduced from 6 to 3)
    
    # Initialize walkers around the best-fit solution
    # Use larger scatter for ensemble analysis
    pos = initial_params + 0.1 * np.random.randn(nwalkers, ndim)
    
    # Ensure all walkers start within prior bounds
    for i in range(nwalkers):
        while log_prior_ensemble(pos[i]) == -np.inf:
            pos[i] = initial_params + 0.1 * np.random.randn(ndim)
    
    # Create sampler for ensemble analysis
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior_ensemble, 
        args=(log_energy, log_age, mass, observed_log_ff, log_ff_errors)
    )
    
    # Run burn-in
    print(f"Running burn-in with {burn_in} steps...")
    pos, _, _ = sampler.run_mcmc(pos, burn_in, progress=True)
    sampler.reset()
    
    # Run production
    print(f"Running MCMC with {nsteps} steps...")
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    # Check convergence
    try:
        tau = sampler.get_autocorr_time()
        print(f"Autocorrelation times: {tau}")
        
        if np.any(tau * 50 > nsteps):
            print("Warning: Chain may not be converged. Consider running longer.")
        else:
            print("Chain appears to be well converged.")
    except Exception as e:
        print(f"Could not compute autocorrelation time: {e}")
        print("This is often due to short chains. Results should still be usable.")
    
    return sampler

def save_samples(samples, param_names, filename='flare_mcmc_samples_fixed_slope.txt'):
    """
    Save MCMC samples to file with header information.
    
    Parameters:
    -----------
    samples : array
        MCMC samples array (n_samples, n_parameters)
    param_names : list
        Names of parameters
    filename : str
        Output filename
    """
    
    print(f"Saving {len(samples)} MCMC samples to {filename}...")
    
    # Create header with parameter information
    header = f"MCMC samples from stellar flare frequency analysis (simplified model)\n"
    header += f"Model: log10(rate) = -1 * log_energy + (b1*log_age + b2*mass + b3)\n"
    header += f"Slope fixed at -1, fitting only intercept parameters\n"
    header += f"Columns: {' '.join(param_names)}\n"
    header += f"Number of samples: {len(samples)}\n"
    header += f"Parameter roles:\n"
    header += f"  b1: Age dependence of normalization\n"
    header += f"  b2: Mass dependence of normalization\n"
    header += f"  b3: Baseline normalization\n"
    
    # Save with numpy
    np.savetxt(filename, samples, 
               header=header,
               fmt='%.6f',
               delimiter='\t')
    
    print(f"Samples saved successfully!")
    print(f"File contains {samples.shape[1]} parameters and {samples.shape[0]} samples")
    
    return filename

def save_samples_multiple_formats(samples, param_names, base_filename='flare_mcmc_samples_fixed_slope'):
    """
    Save MCMC samples in multiple formats for the simplified model.
    
    Parameters:
    -----------
    samples : array
        MCMC samples array
    param_names : list
        Parameter names
    base_filename : str
        Base filename (extensions will be added)
    
    Returns:
    --------
    filenames : dict
        Dictionary of saved filenames
    """
    
    filenames = {}
    
    # 1. Plain text file (easy to read, good for plotting)
    txt_file = f"{base_filename}.txt"
    save_samples(samples, param_names, txt_file)
    filenames['txt'] = txt_file
    
    # 2. CSV file (easy to load in Excel, R, etc.)
    csv_file = f"{base_filename}.csv"
    print(f"Saving samples to {csv_file}...")
    
    import pandas as pd
    df = pd.DataFrame(samples, columns=param_names)
    df.to_csv(csv_file, index=False, float_format='%.6f')
    filenames['csv'] = csv_file
    
    # 3. NumPy binary file (most efficient, preserves full precision)
    npy_file = f"{base_filename}.npy"
    print(f"Saving samples to {npy_file}...")
    np.save(npy_file, samples)
    filenames['npy'] = npy_file
    
    # 4. Save parameter names separately
    params_file = f"{base_filename}_param_names.txt"
    with open(params_file, 'w') as f:
        f.write("# Parameter names for simplified MCMC samples (slope = -1)\n")
        f.write("# Model: log10(rate) = -1 * log_energy + (b1*log_age + b2*mass + b3)\n")
        f.write("# b1, b2, b3: intercept parameters\n")
        for i, name in enumerate(param_names):
            f.write(f"{i}\t{name}\n")
    filenames['params'] = params_file
    
    print(f"\nSaved samples in {len(filenames)} formats:")
    for fmt, filename in filenames.items():
        print(f"  {fmt.upper()}: {filename}")
    
    return filenames
def analyze_results(sampler, thin=10, save_samples_flag=True):
    """
    Analyze MCMC results and create plots for the simplified model.
    
    Parameters:
    -----------
    sampler : emcee.EnsembleSampler
        MCMC sampler with results
    thin : int
        Thinning factor for analysis
    save_samples_flag : bool
        Whether to save the samples to file
        
    Returns:
    --------
    samples : array
        Flattened MCMC samples
    """
    
    # Get samples
    samples = sampler.get_chain(discard=0, thin=thin, flat=True)
    
    # Parameter names for simplified model (slope fixed at -1)
    param_names = ['b1', 'b2', 'b3']
    
    # Save samples to file if requested
    if save_samples_flag:
        filenames = save_samples_multiple_formats(samples, param_names)
    
    # Calculate statistics
    print("\nParameter estimates (median ± 1σ):")
    print("=" * 40)
    print("SIMPLIFIED MODEL: slope = -1 (fixed)")
    print("Model: log10(rate) = -1 * log_energy + (b1*log_age + b2*mass + b3)")
    print("=" * 40)
    
    param_descriptions = [
        'b1: Age dependence of normalization',
        'b2: Mass dependence of normalization',
        'b3: Baseline normalization'
    ]
    
    for i, (name, desc) in enumerate(zip(param_names, param_descriptions)):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{name}: {mcmc[1]:.4f} +{q[1]:.4f} -{q[0]:.4f}  ({desc})")
    
    # Create corner plot
    print("\nCreating corner plot...")
    fig = corner.corner(
        samples, 
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    
    plt.savefig('flare_frequency_fixed_slope_corner.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot chains
    print("Creating trace plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        ax = axes[i]
        ax.plot(sampler.get_chain()[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(sampler.get_chain()))
        ax.set_ylabel(param_names[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("Step number")
    plt.suptitle("MCMC Traces (Simplified Model: slope = -1)", fontsize=14)
    plt.tight_layout()
    plt.savefig('flare_frequency_fixed_slope_traces.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return samples

if __name__ == "__main__":
    # Example usage
    print("Stellar Flare Frequency Ensemble MCMC Analysis (Simplified Model)")
    print("=" * 60)
    print("SIMPLIFIED MODEL: Slope fixed at -1")
    print("Model: log10(rate) = -1 * log_energy + (b1*log_age + b2*mass + b3)")
    print("Fitting only 3 parameters (b1, b2, b3) instead of 6")
    print("=" * 60)
    
    # Run MCMC on the full ensemble dataset
    sampler = run_mcmc_ensemble("ensemble_FFD.csv", nwalkers=32, nsteps=8000, burn_in=1000)
    
    # Analyze results  
    samples = analyze_results(sampler)
    
    print("\nAnalysis complete!")
    print("Output files:")
    print("- flare_frequency_fixed_slope_corner.png: Corner plot with parameter posteriors")
    print("- flare_frequency_fixed_slope_traces.png: MCMC trace plots")
    print("- flare_mcmc_samples_fixed_slope.txt: MCMC samples (tab-separated)")
    print("- flare_mcmc_samples_fixed_slope.csv: MCMC samples (CSV format)")
    print("- flare_mcmc_samples_fixed_slope.npy: MCMC samples (NumPy binary)")
    print("- flare_mcmc_samples_fixed_slope_param_names.txt: Parameter name reference")
    
    # Example of using the fitted model
    print("\nExample: Predicting flare rate for a specific case")
    median_params = np.median(samples, axis=0)
    
    # Note: Simplified model with slope = -1
    # b1, b2, b3 control the normalization (intercept)
    
    # Example stellar parameters
    example_log_energy = 35.0  # log10(erg)
    example_log_age = -0.3     # log10(Gyr) 
    example_mass = 0.7         # solar masses
    
    predicted_log_rate = log_flare_rate_model(
        example_log_energy, example_log_age, example_mass, median_params
    )
    
    print(f"For a star with:")
    print(f"  Mass = {example_mass} M☉")
    print(f"  Age = {10**example_log_age:.1f} Gyr")
    print(f"  Flare energy = 10^{example_log_energy} erg")
    print(f"Predicted flare rate: 10^{predicted_log_rate:.2f} flares/day")
    
    print(f"\nFitted parameters (median values):")
    print("Intercept parameters (slope = -1 fixed):")
    for i, name in enumerate(['b1', 'b2', 'b3']):
        print(f"  {name} = {median_params[i]:.4f}")
    
    print(f"\nModel comparison:")
    print(f"  This simplified model has 3 parameters vs 6 in the full model")
    print(f"  Fixed slope of -1 assumes universal energy distribution")
    print(f"  Only the normalization varies with stellar properties")