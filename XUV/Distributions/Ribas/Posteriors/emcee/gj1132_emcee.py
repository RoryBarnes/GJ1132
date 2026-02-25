#!/usr/bin/env python3
"""
emcee MCMC sampler for GJ 1132 Ribas XUV luminosity evolution model.

Derives posterior distributions for 5 free stellar parameters using
affine-invariant ensemble MCMC with support for informative priors.
"""

import os
import sys
import numpy as np
import emcee
import corner
import multiprocessing
import warnings
import vplot
import time
from collections import deque

# Force single-threaded execution for underlying libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

# Configure matplotlib to avoid LaTeX (prevents package dependency issues)
import matplotlib
matplotlib.rcParams['text.usetex'] = False

import vplanet_inference as vpi
import astropy.units as u

# ===========================================================================
# Configuration
# ===========================================================================

sResultsDir = "results"
if not os.path.exists(sResultsDir):
    os.makedirs(sResultsDir)

iRandomSeed = 42
np.random.seed(iRandomSeed)

# Parameter bounds
dictBounds = {
    "dMass": (0.17, 0.22),
    "dSatXUVFrac": (-4.0, -2.15),  # log10(1e-4) to log10(7e-3)
    "dSatXUVTime": (0.1, 5.0),  # Gyr (will convert to seconds)
    "dXUVBeta": (0.4, 2.1),
    "dStopTime": (1.0, 13.0),  # Gyr (will convert to seconds)
}

listParamNames = [
    "dMass",
    "dSatXUVFrac",
    "dSatXUVTime",
    "dXUVBeta",
    "dStopTime",
]

daParamBounds = np.array([dictBounds[sName] for sName in listParamNames])

# Observational constraints for likelihood
dLBolObs = 4.38e-3
dLBolStd = 3.4e-4
dLogLxuvLbolObs = -4.26
dLogLxuvLbolStd = 0.15

# Informative priors: set to None for uniform, or (mean, std) for symmetric Gaussian
# For asymmetric: (mean, std_upper, std_lower)
# Example: dMassPrior = (0.1945, 0.0048, 0.0046)
dictPriors = {
    "dMass": (0.1945, 0.0048, 0.0046),
    "dSatXUVFrac": (-2.92, 0.26),
    "dSatXUVTime": None,
    "dXUVBeta": (1.18, 0.31),
    "dStopTime": (5.75, 1.38),  # Age in Gyr: 5.75 +1.38/-1.38 Gyr
}

# Prior space: specify whether each prior is in log10 or linear space
# "log" means prior is in log10 space (parameter is converted log10->linear internally)
# "linear" means prior is directly in the parameter space
# Only used if the parameter has a prior (not None)
# Example for log(age): dStopTime is in Gyr (linear), but you have log10(age) prior
dictPriorSpaces = {
    "dMass": "linear",           # Mass prior in Msun (linear)
    "dSatXUVFrac": "log",        # Saturation fraction prior in log10 space
    "dSatXUVTime": "linear",     # Time prior in Gyr (linear)
    "dXUVBeta": "linear",        # Beta prior in linear space
    "dStopTime": "linear",       # Age prior in Gyr (linear, changed from log10)
}

# MCMC configuration
iNWalkers = 100
iNSteps = 1000
iNBurn = 200
iNCores = max(1, multiprocessing.cpu_count() - 1)

# ===========================================================================
# Forward Model
# ===========================================================================

# Global timing tracker for diagnostics
daRecentTimes = deque(maxlen=100)  # Track last 100 model evaluation times
iCallCount = 0

def fnInitVplanetModel():
    inparams = {
        "star.dMass": u.Msun,
        "star.dSatXUVFrac": u.dex(u.dimensionless_unscaled),  # Log10 space
        "star.dSatXUVTime": u.Gyr,  # Linear Gyr
        "star.dXUVBeta": u.dimensionless_unscaled,
        "vpl.dStopTime": u.Gyr,  # Linear Gyr (changed from log10 years)
    }

    outparams = {
        "final.star.Luminosity": u.Lsun,
        "final.star.LXUVStellar": u.Lsun,
    }

    sInpath = os.path.dirname(os.path.abspath(__file__))

    vpm = vpi.VplanetModel(
        inparams, inpath=sInpath, outparams=outparams, verbose=False
    )

    return vpm

try:
    vpm = fnInitVplanetModel()
    print("✓ VplanetModel initialized successfully")
except Exception as e:
    print(f"✗ Error initializing VplanetModel: {e}")
    sys.exit(1)

# ===========================================================================
# Likelihood and Prior Functions
# ===========================================================================

def fdLogLikelihood(daTheta):
    global daRecentTimes, iCallCount

    try:
        # Time the VPlanet model evaluation
        dStartTime = time.time()

        daOutputs = vpm.run_model(daTheta, remove=True)

        dElapsedTime = time.time() - dStartTime
        daRecentTimes.append(dElapsedTime)
        iCallCount += 1

        # Print timing diagnostics every 100 calls
        if iCallCount % 100 == 0:
            dMeanTime = np.mean(daRecentTimes)
            dMedianTime = np.median(daRecentTimes)
            #print(f"\n[Timing] Call {iCallCount}: Mean={dMeanTime:.2f}s, Median={dMedianTime:.2f}s (last 100 calls)")

        # vplanet_inference returns alphabetically sorted: [LXUVStellar, Luminosity]
        dLxuv = daOutputs[0]  # LXUVStellar (alphabetically first)
        dLbol = daOutputs[1]  # Luminosity (alphabetically second)

        # Check for invalid outputs (NaN, zero, negative)
        if not np.isfinite(dLbol) or not np.isfinite(dLxuv):
            return -np.inf
        if dLbol <= 0 or dLxuv <= 0:
            return -np.inf

        dLogRatio = np.log10(dLxuv / dLbol)

        dChi2Lbol = ((dLbol - dLBolObs) / dLBolStd) ** 2
        dChi2Ratio = ((dLogRatio - dLogLxuvLbolObs) / dLogLxuvLbolStd) ** 2

        return -0.5 * (dChi2Lbol + dChi2Ratio)

    except AttributeError as e:
        # VPlanet failed to produce complete output (missing star body)
        # This happens when VPlanet crashes or fails silently
        # Suppress these errors as they're expected for bad parameter combinations
        return -np.inf
    except Exception as e:
        # Log first 10 unexpected errors for debugging
        if iCallCount < 10:
            print(f"\n[Error] VPlanet run failed: {type(e).__name__}: {str(e)[:200]}")
            print(f"  Parameters: {daTheta}")
        return -np.inf


def fdLogPrior(daTheta):
    # Check bounds
    for i, sParamName in enumerate(listParamNames):
        dLower, dUpper = daParamBounds[i]
        if not (dLower <= daTheta[i] <= dUpper):
            # Debug: print first few bound violations
            global iCallCount
            if iCallCount < 5:
                print(f"\n[Debug] Bound violation: {sParamName} = {daTheta[i]:.4f}, bounds = [{dLower}, {dUpper}]")
            return -np.inf

    dLogPrior = 0.0

    # Add informative priors if specified
    # NOTE: MCMC now samples in the SAME space as specified by dictPriorSpaces
    # For "log" priors: MCMC samples in log10 space, prior is in log10 space
    # For "linear" priors: MCMC samples in linear space, prior is in linear space
    # NO conversion or Jacobian needed!
    for i, sParamName in enumerate(listParamNames):
        tPrior = dictPriors[sParamName]

        if tPrior is None:
            continue

        dParam = daTheta[i]  # Already in correct space for prior evaluation

        # Evaluate prior directly (no conversion needed)
        if len(tPrior) == 2:
            dMean, dStd = tPrior
            dLogPrior += -0.5 * ((dParam - dMean) / dStd) ** 2 - np.log(dStd)

        elif len(tPrior) == 3:
            dMean, dStdPos, dStdNeg = tPrior
            if dParam >= dMean:
                dStd = dStdPos
            else:
                dStd = dStdNeg
            dLogPrior += -0.5 * ((dParam - dMean) / dStd) ** 2 - np.log(dStd)

    return dLogPrior


def fdLogProbability(daTheta):
    dLogP = fdLogPrior(daTheta)
    if not np.isfinite(dLogP):
        return -np.inf
    return dLogP + fdLogLikelihood(daTheta)


# ===========================================================================
# MCMC Setup and Execution
# ===========================================================================

def daInitWalkers(iNWalkers, iSeed=None):
    if iSeed is not None:
        np.random.seed(iSeed)

    iNdim = len(listParamNames)
    daPos = np.zeros((iNWalkers, iNdim))

    for i, sParamName in enumerate(listParamNames):
        dLower, dUpper = daParamBounds[i]
        daPos[:, i] = np.random.uniform(dLower, dUpper, iNWalkers)

    return daPos


def fnSamplerRunMcmc():
    iNdim = len(listParamNames)

    print(f"\nInitializing {iNWalkers} walkers...")
    daPos = daInitWalkers(iNWalkers, iSeed=iRandomSeed)

    print(f"Creating multiprocessing pool with {iNCores} cores...")
    with multiprocessing.Pool(processes=iNCores) as pool:
        print("Testing log_probability at initial positions (parallel)...")
        daTestLnprob = np.array(pool.map(fdLogProbability, daPos))
        print(f"  Min log_prob: {np.min(daTestLnprob):.2e}")
        print(f"  Max log_prob: {np.max(daTestLnprob):.2e}")
        print(f"  Finite: {np.sum(np.isfinite(daTestLnprob))}/{iNWalkers}")

        print(f"\nCreating EnsembleSampler...")
        sampler = emcee.EnsembleSampler(
            iNWalkers,
            iNdim,
            fdLogProbability,
            pool=pool,
            moves=emcee.moves.StretchMove(),
        )

        print(f"\nRunning burn-in phase ({iNBurn} steps)...")
        state = sampler.run_mcmc(daPos, iNBurn, progress=True)
        print(f"✓ Burn-in complete")
        print(f"  Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

        sampler.reset()

        print(f"\nRunning production phase ({iNSteps} steps)...")
        sampler.run_mcmc(state, iNSteps, progress=True)
        print(f"✓ Production phase complete")

    return sampler


def fnAnalyzeChain(sampler):
    fAcceptFrac = np.mean(sampler.acceptance_fraction)
    #print(f"\n{'='*70}")
    print(f"MCMC Diagnostics")
    print(f"{'='*70}")
    print(f"Mean acceptance fraction: {fAcceptFrac:.3f}")

    if 0.2 < fAcceptFrac < 0.5:
        print("  ✓ Acceptance fraction in good range [0.2-0.5]")
    elif fAcceptFrac < 0.2:
        print("  ⚠ Acceptance fraction low (<0.2)")
    else:
        print("  ⚠ Acceptance fraction high (>0.5)")

    try:
        daAuto = sampler.get_autocorr_time(quiet=True)
        print(f"\nAutocorrelation times:")
        for i, sName in enumerate(listParamNames):
            print(f"  {sName:15s}: {daAuto[i]:8.1f} steps")

        if np.all(daAuto < 0.1 * iNSteps):
            print("  ✓ Autocorr times < 10% of chain length")
        else:
            print("  ⚠ Some parameters have long autocorr times")

    except Exception as e:
        print(f"  Could not compute autocorr time: {e}")

    print(f"\nFlattening chains...")
    daSamples = sampler.get_chain(flat=True)
    print(f"  Total samples: {daSamples.shape[0]}")
    print(f"  Walkers: {sampler.nwalkers}, Steps: {sampler.iteration}")

    return daSamples


def fnSaveResults(sampler, daSamples):
    print(f"\nSaving results to {sResultsDir}/...")

    sChainFile = os.path.join(sResultsDir, "emcee_chains.npz")
    np.savez(
        sChainFile,
        chain=sampler.get_chain(),
        log_prob=sampler.get_log_prob(),
        samples=daSamples,
        param_names=listParamNames,
    )
    print(f"  ✓ {sChainFile}")

    sSamplesFile = os.path.join(sResultsDir, "emcee_samples_flat.npy")
    np.save(sSamplesFile, daSamples)
    print(f"  ✓ {sSamplesFile}")

    sInfoFile = os.path.join(sResultsDir, "emcee_params.txt")
    with open(sInfoFile, "w") as f:
        f.write("Parameter Information\n")
        f.write("=" * 70 + "\n\n")
        f.write("Parameter Bounds:\n")
        for i, sName in enumerate(listParamNames):
            dLower, dUpper = daParamBounds[i]
            f.write(f"  {sName:15s}: [{dLower:10.6f}, {dUpper:10.6f}]\n")
        f.write("\nPriors:\n")
        for sName in listParamNames:
            if dictPriors[sName] is None:
                f.write(f"  {sName:15s}: Uniform\n")
            else:
                sPriorSpace = dictPriorSpaces[sName]
                f.write(f"  {sName:15s}: {dictPriors[sName]} [{sPriorSpace}]\n")
        f.write("\nObservational Constraints:\n")
        f.write(f"  L_bol:            {dLBolObs:.6e} ± {dLBolStd:.6e} Lsun\n")
        f.write(f"  log(L_XUV/L_bol): {dLogLxuvLbolObs:.3f} ± {dLogLxuvLbolStd:.3f}\n")
        f.write("\nMCMC Configuration:\n")
        f.write(f"  iNWalkers: {iNWalkers}\n")
        f.write(f"  iNSteps:   {iNSteps}\n")
        f.write(f"  iNBurn:    {iNBurn}\n")
        f.write(f"  iNCores:   {iNCores}\n")
    print(f"  ✓ {sInfoFile}")


def fnMakeCornerPlot(daSamples):
    print(f"\nGenerating corner plot...")

    listLabels = [
        r"$M_{\star}$ [$M_{\odot}$]",
        r"$\log_{10}(f_{sat})$",
        r"$t_{sat}$ [Gyr]",
        r"$\beta_{XUV}$",
        r"Age [Gyr]",
    ]

    fig = corner.corner(
        daSamples,
        labels=listLabels,
        range=daParamBounds,
        bins=30,
        hist_kwargs={"density": True},
        label_kwargs={"fontsize": 12},
    )

    # Save both PNG and PDF
    sPngFile = os.path.join(sResultsDir, "emcee_corner.png")
    fig.savefig(sPngFile, dpi=300, bbox_inches="tight")
    print(f"  ✓ {sPngFile}")

    sPlotFile = os.path.join(sResultsDir, "emcee_corner.pdf")
    fig.savefig(sPlotFile, dpi=300, bbox_inches="tight")
    print(f"  ✓ {sPlotFile}")


# ===========================================================================
# Main Execution
# ===========================================================================

if __name__ == "__main__":
    # print("\n" + "=" * 70)
    # print("GJ 1132 Ribas XUV Luminosity Evolution - emcee MCMC")
    # print("=" * 70)

    print("\nStarting MCMC sampling...")
    sampler = fnSamplerRunMcmc()

    daSamples = fnAnalyzeChain(sampler)

    fnSaveResults(sampler, daSamples)

    fnMakeCornerPlot(daSamples)

    #print(f"\n{'='*70}")
    print("MCMC sampling completed successfully!")
    #print(f"{'='*70}\n")
