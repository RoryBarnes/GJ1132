#!/usr/bin/env python3
"""
Test version of emcee MCMC sampler with reduced iterations.

This is identical to gj1132_emcee.py but with N_STEPS and N_BURN reduced
for quick testing without waiting for convergence.
"""

import os
import sys
import numpy as np
import emcee
import corner
import multiprocessing
from functools import partial
import warnings

# Force single-threaded execution for underlying libraries to prevent nested threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure matplotlib to avoid LaTeX (prevents "latex not found" errors)
import matplotlib
matplotlib.rcParams['text.usetex'] = False

# Import vplanet modules
import vplanet_inference as vpi
import astropy.units as u

# ===========================================================================
# Configuration (REDUCED FOR TESTING)
# ===========================================================================

# Output directory
RESULTS_DIR = "results_test"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Parameter bounds (empirically derived from dynesty posteriors)
BOUNDS = {
    "dMass": (0.15, 0.22),              # Msun
    "dSatXUVFrac": (1e-4, 7e-3),        # linear scale
    "dSatXUVTime": (0.1, 5.0),          # Gyr
    "dXUVBeta": (0.4, 2.1),             # dimensionless
    "dStopTime": (3.6, 13.1),           # Gyr
}

# Parameter names (in order for theta vector)
PARAM_NAMES = [
    "dMass",
    "dSatXUVFrac",
    "dSatXUVTime",
    "dXUVBeta",
    "dStopTime",
]

# Bounds as array for prior checks
PARAM_BOUNDS = np.array([BOUNDS[name] for name in PARAM_NAMES])

# Observational constraints
L_BOL_OBS = 4.38e-3  # Lsun
L_BOL_STD = 3.4e-4   # Lsun

LOG_LXUV_LBOL_OBS = -4.26
LOG_LXUV_LBOL_STD = 0.15

# MCMC configuration (REDUCED FOR TESTING)
N_WALKERS = 20          # Reduced from 100
N_STEPS = 200           # Reduced from 50000
N_BURN = 50             # Reduced from 10000

# Number of cores for parallel execution
N_CORES = max(1, multiprocessing.cpu_count() - 1)

# ===========================================================================
# Forward Model Initialization
# ===========================================================================

def init_vplanet_model():
    """Initialize the VplanetModel for forward modeling."""
    inparams = {
        "star.dMass": u.Msun,
        "star.dSatXUVFrac": u.dex(u.dimensionless_unscaled),
        "star.dSatXUVTime": u.Gyr,
        "star.dXUVBeta": u.dimensionless_unscaled,
        "vpl.dStopTime": u.Gyr,
    }

    outparams = {
        "final.star.Luminosity": u.Lsun,
        "final.star.LXUVStellar": u.Lsun,
    }

    inpath = os.path.dirname(os.path.abspath(__file__))

    vpm = vpi.VplanetModel(
        inparams,
        inpath=inpath,
        outparams=outparams,
        verbose=False,
    )

    return vpm


# Global VplanetModel instance
try:
    VPM = init_vplanet_model()
    print("✓ VplanetModel initialized successfully")
except Exception as e:
    print(f"✗ Error initializing VplanetModel: {e}")
    sys.exit(1)


# ===========================================================================
# Likelihood and Prior Functions
# ===========================================================================

def log_likelihood(theta):
    """Compute log-likelihood for given model parameters."""
    try:
        outputs = VPM.run_model(theta, remove=True)

        lbol = outputs[0]
        lxuv = outputs[1]

        if lbol <= 0 or lxuv <= 0:
            return -np.inf

        log_lxuv_lbol = np.log10(lxuv / lbol)

        chi2_lbol = ((lbol - L_BOL_OBS) / L_BOL_STD) ** 2
        chi2_lxuv = ((log_lxuv_lbol - LOG_LXUV_LBOL_OBS) / LOG_LXUV_LBOL_STD) ** 2
        chi2_total = chi2_lbol + chi2_lxuv

        lnl = -0.5 * chi2_total

        return lnl

    except Exception as e:
        return -np.inf


def log_prior(theta):
    """Compute log-prior (uniform bounds)."""
    for i, param_name in enumerate(PARAM_NAMES):
        lower, upper = PARAM_BOUNDS[i]
        if not (lower <= theta[i] <= upper):
            return -np.inf

    return 0.0


def log_probability(theta):
    """Compute log-probability (prior + likelihood)."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta)


# ===========================================================================
# MCMC Initialization and Sampling
# ===========================================================================

def init_walkers(nwalkers, seed=None):
    """Initialize walker positions randomly within bounds."""
    if seed is not None:
        np.random.seed(seed)

    ndim = len(PARAM_NAMES)
    pos = np.zeros((nwalkers, ndim))

    for i, param_name in enumerate(PARAM_NAMES):
        lower, upper = PARAM_BOUNDS[i]
        pos[:, i] = np.random.uniform(lower, upper, nwalkers)

    return pos


def run_mcmc():
    """Run the MCMC sampler."""
    ndim = len(PARAM_NAMES)

    print(f"\nInitializing {N_WALKERS} walkers...")
    pos = init_walkers(N_WALKERS, seed=RANDOM_SEED)

    print(f"Creating multiprocessing pool with {N_CORES} cores...")
    with multiprocessing.Pool(processes=N_CORES) as pool:
        print("Testing log_probability at initial positions (parallel)...")
        test_lnprob = np.array(pool.map(log_probability, pos))
        if not any(np.isfinite(test_lnprob)):
            print("✗ Warning: No finite log-probability at initial positions.")

        print(f"  Min log_prob: {np.min(test_lnprob):.2e}")
        print(f"  Max log_prob: {np.max(test_lnprob):.2e}")
        print(f"  Finite values: {np.sum(np.isfinite(test_lnprob))}/{N_WALKERS}")

        print(f"\nCreating EnsembleSampler...")
        sampler = emcee.EnsembleSampler(
            N_WALKERS,
            ndim,
            log_probability,
            pool=pool,
            moves=emcee.moves.StretchMove(),
        )

        print(f"\nRunning burn-in phase ({N_BURN} steps)...")
        state = sampler.run_mcmc(pos, N_BURN, progress=True)
        print(f"✓ Burn-in complete")
        print(f"  Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

        sampler.reset()

        print(f"\nRunning production phase ({N_STEPS} steps)...")
        sampler.run_mcmc(state, N_STEPS, progress=True)
        print(f"✓ Production phase complete")

    return sampler


# ===========================================================================
# Post-Processing and Diagnostics
# ===========================================================================

def analyze_chain(sampler):
    """Compute diagnostics and generate posterior samples."""
    acceptance_fraction = np.mean(sampler.acceptance_fraction)
    print(f"\n{'='*70}")
    print(f"MCMC Diagnostics")
    print(f"{'='*70}")
    print(f"Mean acceptance fraction: {acceptance_fraction:.3f}")

    if 0.2 < acceptance_fraction < 0.5:
        print("  ✓ Acceptance fraction in good range [0.2-0.5]")
    elif acceptance_fraction < 0.2:
        print("  ⚠ Acceptance fraction low (<0.2), may need tuning")
    elif acceptance_fraction > 0.5:
        print("  ⚠ Acceptance fraction high (>0.5), may need tuning")

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"\nAutocorrelation times:")
        for i, param_name in enumerate(PARAM_NAMES):
            print(f"  {param_name:15s}: {tau[i]:8.1f} steps")

        if np.all(tau < 0.1 * N_STEPS):
            print("  ✓ Autocorrelation times < 10% of chain length")
        else:
            print("  ⚠ Some parameters have long autocorrelation times")

    except Exception as e:
        print(f"  Could not compute autocorrelation time: {e}")

    print(f"\nFlattening chains...")
    samples = sampler.get_chain(flat=True)
    print(f"  Total samples: {samples.shape[0]}")
    print(f"  Walkers: {sampler.nwalkers}, Steps: {sampler.iteration}")

    return samples


def save_results(sampler, samples):
    """Save chains and posterior samples to files."""
    print(f"\nSaving results to {RESULTS_DIR}/...")

    chains_file = os.path.join(RESULTS_DIR, "emcee_chains.npz")
    np.savez(
        chains_file,
        chain=sampler.get_chain(),
        log_prob=sampler.get_log_prob(),
        samples=samples,
        param_names=PARAM_NAMES,
    )
    print(f"  ✓ {chains_file}")

    samples_file = os.path.join(RESULTS_DIR, "emcee_samples_flat.npy")
    np.save(samples_file, samples)
    print(f"  ✓ {samples_file}")


def make_corner_plot(samples):
    """Generate and save corner plot of posterior distributions."""
    print(f"\nGenerating corner plot...")

    labels = [
        r"$M_{\star}$ [$M_{\odot}$]",
        r"$f_{sat}$ [XUV]",
        r"$t_{sat}$ [Gyr]",
        r"$\beta_{XUV}$",
        r"Age [Gyr]",
    ]

    fig = corner.corner(
        samples,
        labels=labels,
        range=PARAM_BOUNDS,
        bins=20,
        hist_kwargs={"density": True},
        label_kwargs={"fontsize": 10},
    )

    plot_file = os.path.join(RESULTS_DIR, "emcee_corner.png")
    fig.savefig(plot_file, dpi=100, bbox_inches="tight")
    print(f"  ✓ {plot_file}")


# ===========================================================================
# Main Execution
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GJ 1132 Ribas XUV Luminosity Evolution - emcee MCMC (TEST)")
    print("=" * 70)

    print("\nStarting MCMC sampling...")
    sampler = run_mcmc()

    samples = analyze_chain(sampler)

    save_results(sampler, samples)

    make_corner_plot(samples)

    print(f"\n{'='*70}")
    print("Test MCMC completed!")
    print(f"{'='*70}\n")
