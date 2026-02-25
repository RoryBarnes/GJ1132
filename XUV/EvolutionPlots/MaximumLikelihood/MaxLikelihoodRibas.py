#!/usr/bin/env python3
"""
Maximum likelihood estimation for GJ 1132 Ribas XUV luminosity evolution model.

Finds the parameter values that maximize the likelihood given observational
constraints on L_bol and log(L_XUV/L_bol), then generates an evolution plot
comparing L_XUV/L_bol as a function of time.
"""

import os
import sys
import numpy as np
from scipy.optimize import differential_evolution
import warnings

# Import matplotlib and vplot in correct order
# vplot must be imported BEFORE matplotlib.pyplot to set font styling
import matplotlib
import vplot

# Configure matplotlib to NEVER use LaTeX
# Must be set AFTER vplot import in case vplot tries to enable it
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.preamble'] = ''
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
matplotlib.rcParams['font.size'] = 12

# Now import pyplot after vplot has configured fonts
import matplotlib.pyplot as plt

# Add parent directory to path for vplanet imports
# From /XUV/EvolutionPlots/MaximumLikelihood -> /XUV/Distributions/Ribas/Posteriors/emcee
sScriptDir = os.path.dirname(os.path.abspath(__file__))
sXUVDir = os.path.dirname(os.path.dirname(sScriptDir))
sSrcPath = os.path.join(sXUVDir, "Distributions/Ribas/Posteriors/alabi")
sys.path.insert(0, sSrcPath)

warnings.filterwarnings("ignore")

# Force single-threaded execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import vplanet_inference as vpi
import astropy.units as u

# ===========================================================================
# Configuration
# ===========================================================================

# Parameter bounds (from emcee script)
dictBounds = {
    "dMass": (0.17, 0.22),
    "dSatXUVFrac": (-4.0, -2.15),  # log10(1e-4) to log10(7e-3)
    "dSatXUVTime": (0.1, 5.0),  # Gyr
    "dXUVBeta": (0.4, 2.1),
    "dStopTime": (1.0, 13.0),  # Gyr
}

listParamNames = [
    "dMass",
    "dSatXUVFrac",
    "dSatXUVTime",
    "dXUVBeta",
    "dStopTime",
]

daParamBounds = np.array([dictBounds[sName] for sName in listParamNames])

# Observational constraints
dLBolObs = 4.38e-3
dLBolStd = 3.4e-4
dLogLxuvLbolObs = -4.26
dLogLxuvLbolStd = 0.15

# Informative priors (from literature, same as emcee script)
# Set to None for uniform, or (mean, std) for Gaussian
# For asymmetric: (mean, std_upper, std_lower)
dictPriors = {
    "dMass": (0.1945, 0.0048, 0.0046),  # Asymmetric Gaussian from observations
    "dSatXUVFrac": (-2.92, 0.26),       # Log10 space Gaussian from XUV studies
    "dSatXUVTime": None,                # No prior - poorly constrained
    "dXUVBeta": (1.18, 0.31),           # Gaussian from Ribas et al.
    "dStopTime": (5.75, 1.38),          # Gaussian from age constraints (Gyr)
}

# Prior space specification
dictPriorSpaces = {
    "dMass": "linear",       # Mass prior in Msun (linear)
    "dSatXUVFrac": "log",    # Saturation fraction prior in log10 space
    "dSatXUVTime": "linear", # Time prior in Gyr (linear)
    "dXUVBeta": "linear",    # Beta prior in linear space
    "dStopTime": "linear",   # Age prior in Gyr (linear)
}

# ===========================================================================
# VPlanet Model Setup
# ===========================================================================

def fnInitVplanetModel():
    """Initialize VPlanet model for GJ 1132 system."""
    inparams = {
        "star.dMass": u.Msun,
        "star.dSatXUVFrac": u.dex(u.dimensionless_unscaled),  # Log10 space
        "star.dSatXUVTime": u.Gyr,  # Linear Gyr
        "star.dXUVBeta": u.dimensionless_unscaled,
        "vpl.dStopTime": u.Gyr,  # Linear Gyr
    }

    outparams = {
        "final.star.Luminosity": u.Lsun,
        "final.star.LXUVStellar": u.Lsun,
    }

    vpm = vpi.VplanetModel(
        inparams, inpath=sSrcPath, outparams=outparams, verbose=False
    )

    return vpm

try:
    vpm = fnInitVplanetModel()
    print("✓ VplanetModel initialized")
except Exception as e:
    print(f"✗ Error initializing VplanetModel: {e}")
    sys.exit(1)

# ===========================================================================
# Likelihood and Prior Functions
# ===========================================================================

def fdNegLogPrior(daTheta):
    """Calculate negative log-prior for optimization."""
    dNegLogPrior = 0.0

    # Add informative priors if specified
    # NOTE: For this optimization, parameters are sampled in the SAME space
    # as specified in dictPriorSpaces. NO conversion or Jacobian needed.
    # For "log" priors: parameter is in log10 space, prior is in log10 space
    # For "linear" priors: parameter is in linear space, prior is in linear space
    for i, sParamName in enumerate(listParamNames):
        tPrior = dictPriors[sParamName]

        if tPrior is None:
            continue

        dParam = daTheta[i]  # Already in correct space for prior evaluation

        # Evaluate prior directly (no conversion needed)
        if len(tPrior) == 2:
            # Symmetric Gaussian
            dMean, dStd = tPrior
            dNegLogPrior += 0.5 * ((dParam - dMean) / dStd) ** 2
        elif len(tPrior) == 3:
            # Asymmetric Gaussian
            dMean, dStdPos, dStdNeg = tPrior
            if dParam >= dMean:
                dStd = dStdPos
            else:
                dStd = dStdNeg
            dNegLogPrior += 0.5 * ((dParam - dMean) / dStd) ** 2

    return dNegLogPrior


def fdNegLogLikelihood(daTheta):
    """Calculate negative log-likelihood (data only, no priors)."""
    try:
        # Check bounds
        for i in range(len(daTheta)):
            if not (daParamBounds[i, 0] <= daTheta[i] <= daParamBounds[i, 1]):
                return 1e10

        daOutputs = vpm.run_model(daTheta, remove=True)

        # vplanet_inference returns alphabetically sorted: [LXUVStellar, Luminosity]
        dLxuv = daOutputs[0]  # LXUVStellar (alphabetically first)
        dLbol = daOutputs[1]  # Luminosity (alphabetically second)

        # Check for invalid outputs (NaN, zero, negative)
        if not np.isfinite(dLbol) or not np.isfinite(dLxuv):
            return 1e10
        if dLbol <= 0 or dLxuv <= 0:
            return 1e10

        dLogRatio = np.log10(dLxuv / dLbol)

        dChi2Lbol = ((dLbol - dLBolObs) / dLBolStd) ** 2
        dChi2Ratio = ((dLogRatio - dLogLxuvLbolObs) / dLogLxuvLbolStd) ** 2

        # Return negative log-likelihood (minimize instead of maximize)
        return 0.5 * (dChi2Lbol + dChi2Ratio)

    except AttributeError:
        # VPlanet failed to produce complete output (missing star body)
        return 1e10
    except Exception:
        return 1e10


def fdNegLogPosterior(daTheta):
    """Calculate negative log-posterior (likelihood + prior)."""
    # Check bounds first
    for i in range(len(daTheta)):
        if not (daParamBounds[i, 0] <= daTheta[i] <= daParamBounds[i, 1]):
            return 1e10

    dNegLogPrior = fdNegLogPrior(daTheta)
    dNegLogLike = fdNegLogLikelihood(daTheta)

    # Return sum (posterior = likelihood × prior in probability space)
    return dNegLogLike + dNegLogPrior

# ===========================================================================
# Maximum Likelihood Optimization
# ===========================================================================

def daFindMaxLikelihood():
    """Find maximum a posteriori (MAP) parameters using differential evolution."""
    print("\nStarting maximum a posteriori (MAP) optimization...")
    print("Method: Differential Evolution")
    print(f"Parameter space: {len(listParamNames)} dimensions")

    # Print bounds for verification
    print("\nParameter bounds:")
    for i, sName in enumerate(listParamNames):
        print(f"  {sName:15s}: [{daParamBounds[i,0]:10.6f}, {daParamBounds[i,1]:10.6f}]")

    # Degeneracy detection: track optimization progress
    listCallbackHistory = []
    iStagnationCount = 0
    iMaxStagnation = 8  # Warn after 8 steps with no improvement

    def fnCheckStagnation(daXk, convergence):
        """Callback to detect optimization stagnation (indicates degeneracy)."""
        nonlocal iStagnationCount

        # Get current best value
        dCurrentBest = fdNegLogPosterior(daXk)
        listCallbackHistory.append(dCurrentBest)

        # Check for stagnation (no improvement over last few iterations)
        if len(listCallbackHistory) > 3:
            dRecentBest = min(listCallbackHistory[-4:])
            dImprovementRatio = abs(dCurrentBest - dRecentBest) / (abs(dRecentBest) + 1e-10)

            if dImprovementRatio < 1e-4:  # Less than 0.01% improvement
                iStagnationCount += 1
                if iStagnationCount >= iMaxStagnation:
                    print(f"\n⚠ WARNING: Optimization stagnating (no improvement for {iStagnationCount} steps)")
                    print("  This may indicate parameter degeneracy or poor constraints.")
                    print("  Current -ln(Posterior) = {:.6f}".format(dCurrentBest))
            else:
                iStagnationCount = 0  # Reset if we see improvement

        return False  # Don't stop, just warn

    # Print priors being used
    print("\nInformative priors:")
    for i, sName in enumerate(listParamNames):
        tPrior = dictPriors[sName]
        if tPrior is None:
            print(f"  {sName:15s}: Uniform (no prior)")
        else:
            sPriorSpace = dictPriorSpaces[sName]
            if len(tPrior) == 2:
                print(f"  {sName:15s}: Gaussian({tPrior[0]:.3f}, {tPrior[1]:.3f}) [{sPriorSpace}]")
            else:
                print(f"  {sName:15s}: Gaussian({tPrior[0]:.3f}, +{tPrior[1]:.3f}/-{tPrior[2]:.3f}) [{sPriorSpace}]")

    result = differential_evolution(
        fdNegLogPosterior,  # Optimize posterior (likelihood + priors)
        daParamBounds,
        strategy='best1bin',
        maxiter=50,          # Reduced from 100 for faster convergence
        popsize=10,          # Reduced from 15 for faster convergence
        tol=0.01,            # Relative tolerance (fraction of population std)
        atol=1e-4,           # Absolute tolerance for f(x) convergence
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        workers=1,
        updating='deferred',
        polish=False,  # Disable polish to strictly enforce bounds
        callback=fnCheckStagnation,  # Detect degeneracy during optimization
        disp=True
    )

    if result.success:
        print("\n✓ Optimization converged successfully")
    else:
        print("\n⚠ Optimization did not fully converge")
        print(f"Message: {result.message}")

    # Verify final parameters are in bounds
    print("\nFinal parameters:")
    for i, sName in enumerate(listParamNames):
        dVal = result.x[i]
        bInBounds = daParamBounds[i,0] <= dVal <= daParamBounds[i,1]
        sStatus = "✓" if bInBounds else "✗ OUT OF BOUNDS"
        print(f"  {sName:15s} = {dVal:12.6e} {sStatus}")

    # DEGENERACY DIAGNOSTICS: Check if optimization was meaningful
    print("\n" + "="*70)
    print("DEGENERACY DIAGNOSTICS")
    print("="*70)

    # Decompose final posterior
    dFinalNegLogPrior = fdNegLogPrior(result.x)
    dFinalNegLogLike = fdNegLogLikelihood(result.x)
    dFinalNegLogPost = result.fun

    print(f"\nFinal -ln(Posterior): {dFinalNegLogPost:.6f}")
    print(f"  -ln(Likelihood):    {dFinalNegLogLike:.6f}")
    print(f"  -ln(Prior):         {dFinalNegLogPrior:.6f}")

    # Context: underdetermined problem
    iNumConstraints = 2  # L_bol and log(L_XUV/L_bol)
    iNumFreeParams = len(listParamNames)
    print(f"\nProblem characteristics:")
    print(f"  Number of observational constraints: {iNumConstraints}")
    print(f"  Number of free parameters:           {iNumFreeParams}")
    if iNumFreeParams > iNumConstraints:
        print(f"  ⚠ UNDERDETERMINED SYSTEM ({iNumFreeParams} params > {iNumConstraints} constraints)")
        print(f"  Low χ² is EXPECTED - model can fit data almost perfectly!")
        print(f"  Priors are essential to select physically reasonable solution.")

    # Convert to more intuitive metrics
    dChi2 = 2.0 * dFinalNegLogLike  # χ² = 2 * (-ln(L)) for Gaussian likelihoods
    print(f"\nFit quality:")
    print(f"  χ² = {dChi2:.6f}  (for {iNumConstraints} constraints)")
    if iNumConstraints > 0:
        print(f"  χ²/N_obs = {dChi2/iNumConstraints:.6f}")
        print(f"  → Model fits data to {np.sqrt(dChi2/iNumConstraints):.3f}σ per observable")


    # Check balance between likelihood and prior
    print("\nLikelihood vs Prior balance:")
    if dFinalNegLogLike < dFinalNegLogPrior / 10.0:
        print("  ⚠ Likelihood << Prior: Fit is prior-dominated")
        print("    → Data is weakly informative, posteriors ≈ priors")
        print("    → MCMC will mainly sample from priors")
    elif dFinalNegLogPrior < dFinalNegLogLike / 10.0:
        print("  ⚠ Prior << Likelihood: Fit is data-dominated")
        print("    → Parameters may have drifted far from prior expectations")
        print("    → Check if this is physically reasonable")
    else:
        print("  ✓ Balanced: Both data and priors contribute meaningfully")
        print("    → MCMC posteriors will be informative compromises")

    if dFinalNegLogPrior > 10.0:
        print("\n⚠ WARNING: -ln(Prior) is large (> 10)")
        print("  Final parameters are far from prior expectations.")
        print("  Check if priors are appropriate or if data contradicts assumptions.")

    # Check parameter movement from prior means
    print("\nParameter deviations from prior means:")
    bLargeDrift = False
    for i, sName in enumerate(listParamNames):
        tPrior = dictPriors[sName]
        if tPrior is not None:
            dFinalValue = result.x[i]
            if len(tPrior) == 2:
                dMean, dStd = tPrior
            else:
                dMean, dStdPos, dStdNeg = tPrior
                dStd = (dStdPos + dStdNeg) / 2.0

            dSigmas = abs(dFinalValue - dMean) / dStd
            sWarning = ""
            if dSigmas > 3.0:
                sWarning = " ⚠ >3σ from prior!"
                bLargeDrift = True
            print(f"  {sName:15s}: {dSigmas:5.2f}σ from prior{sWarning}")

    if bLargeDrift:
        print("\n⚠ Some parameters drifted >3σ from prior means.")
        print("  This suggests either:")
        print("    (1) Data strongly constrains these parameters differently than priors")
        print("    (2) Priors were poorly chosen")
        print("    (3) Degeneracy allows wandering despite priors")

    # MCMC viability assessment
    print("\n" + "="*70)
    print("MCMC VIABILITY ASSESSMENT")
    print("="*70)

    if iStagnationCount > 0:
        print(f"⚠ Optimization showed stagnation ({iStagnationCount} stagnant steps at end)")
        print("  MCMC may struggle with this problem due to parameter degeneracy.")
        print("  Recommend: Add stronger priors or fix poorly constrained parameters.")
    else:
        print("✓ Optimization converged without significant stagnation")
        print("  MCMC should be viable for this problem.")

    if dFinalNegLogLike < 1.0 and not bLargeDrift:
        print("✓ Good fit achieved without excessive parameter drift")
        print("  MCMC posteriors should be well-behaved.")

    print("="*70 + "\n")

    return result.x, result.fun

# ===========================================================================
# Evolution Calculation
# ===========================================================================

def fdaGetEvolution(daParams, daTimeGyr):
    """Calculate L_XUV/L_bol evolution over time."""
    daLxuvLbol = np.zeros_like(daTimeGyr)

    for i, dTime in enumerate(daTimeGyr):
        daTheta = daParams.copy()
        daTheta[4] = dTime  # Update dStopTime

        try:
            daOutputs = vpm.run_model(daTheta, remove=True)
            # vplanet_inference returns alphabetically sorted: [LXUVStellar, Luminosity]
            dLxuv = daOutputs[0]  # LXUVStellar
            dLbol = daOutputs[1]  # Luminosity

            if np.isfinite(dLbol) and np.isfinite(dLxuv) and dLbol > 0 and dLxuv > 0:
                daLxuvLbol[i] = dLxuv / dLbol
            else:
                daLxuvLbol[i] = np.nan
        except Exception:
            daLxuvLbol[i] = np.nan

    return daLxuvLbol

# ===========================================================================
# Plotting
# ===========================================================================

def fbPlotEvolution(daParams, dNegLogLike):
    """Generate evolution plot of L_XUV/L_bol vs time."""
    print("\nGenerating evolution plot...")

    # Ensure LaTeX is disabled for this plot
    plt.rcParams['text.usetex'] = False

    # Time grid from 100 Myr to 13 Gyr
    daTimeGyr = np.logspace(np.log10(0.1), np.log10(13.0), 100)

    # Calculate evolution
    print("Computing L_XUV/L_bol evolution...")
    daLxuvLbol = fdaGetEvolution(daParams, daTimeGyr)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot maximum likelihood evolution
    ax.loglog(daTimeGyr, daLxuvLbol, 'b-', linewidth=2,
              label='Maximum Likelihood Model')

    # Add observational constraint
    dAgeGyr = daParams[4]  # System age from max likelihood
    dLxuvLbolObs = 10**dLogLxuvLbolObs
    dLxuvLbolErrLower = 10**(dLogLxuvLbolObs - dLogLxuvLbolStd)
    dLxuvLbolErrUpper = 10**(dLogLxuvLbolObs + dLogLxuvLbolStd)

    ax.errorbar(
        dAgeGyr, dLxuvLbolObs,
        yerr=[[dLxuvLbolObs - dLxuvLbolErrLower], [dLxuvLbolErrUpper - dLxuvLbolObs]],
        fmt='ro', markersize=8, capsize=5, capthick=2,
        label='Observed (GJ 1132)', zorder=10
    )

    # Format plot - use mathtext (not LaTeX) for symbols
    ax.set_xlabel('Age [Gyr]', fontsize=14)
    ax.set_ylabel('L_XUV / L_bol', fontsize=14)  # Plain text, no math mode
    ax.set_title('GJ 1132: XUV Evolution (Maximum Likelihood)', fontsize=16)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12, loc='best')

    # Set reasonable axis limits
    ax.set_xlim(0.08, 15)
    ax.set_ylim(1e-6, 1e-2)

    # Add parameter text box - use plain text to avoid LaTeX
    sParamText = 'Maximum Likelihood Parameters:\n'
    sParamText += f'M_star = {daParams[0]:.4f} M_sun\n'
    sParamText += f'log10(f_sat) = {daParams[1]:.3f}\n'
    sParamText += f't_sat = {daParams[2]:.3f} Gyr\n'
    sParamText += f'beta_XUV = {daParams[3]:.3f}\n'
    sParamText += f'Age = {daParams[4]:.3f} Gyr\n'
    sParamText += f'-ln(L) = {dNegLogLike:.3f}'

    ax.text(0.02, 0.98, sParamText, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save figure with explicit backend to avoid LaTeX issues
    sPlotFile = 'gj1132_maxlike_evolution.pdf'
    fig.savefig(sPlotFile, dpi=300, bbox_inches='tight', backend='pdf')
    print(f"✓ Plot saved: {sPlotFile}")

    return True

# ===========================================================================
# Results Output
# ===========================================================================

def fbSaveResults(daParams, dNegLogPost):
    """Save maximum a posteriori (MAP) results to file."""
    sResultFile = 'maxlike_results.txt'

    # Calculate individual components
    dNegLogPrior = fdNegLogPrior(daParams)
    dNegLogLike = fdNegLogLikelihood(daParams)

    with open(sResultFile, 'w') as f:
        f.write("GJ 1132 Maximum A Posteriori (MAP) Estimation\n")
        f.write("=" * 70 + "\n\n")
        f.write("Ribas XUV Evolution Model\n\n")

        f.write("Maximum A Posteriori (MAP) Parameters:\n")
        f.write("-" * 70 + "\n")
        for i, sName in enumerate(listParamNames):
            f.write(f"{sName:15s} = {daParams[i]:.6e}\n")

        f.write(f"\n-ln(Posterior)  = {dNegLogPost:.6e}  (minimized objective)\n")
        f.write(f"-ln(Likelihood) = {dNegLogLike:.6e}\n")
        f.write(f"-ln(Prior)      = {dNegLogPrior:.6e}\n")
        f.write(f"chi^2           = {2*dNegLogLike:.6e}\n")

        f.write("\nObservational Constraints:\n")
        f.write("-" * 70 + "\n")
        f.write(f"L_bol (obs)            = {dLBolObs:.6e} ± {dLBolStd:.6e} Lsun\n")
        f.write(f"log(L_XUV/L_bol) (obs) = {dLogLxuvLbolObs:.3f} ± {dLogLxuvLbolStd:.3f}\n")

        # Calculate model predictions at max likelihood
        try:
            daOutputs = vpm.run_model(daParams, remove=True)
            # vplanet_inference returns alphabetically sorted: [LXUVStellar, Luminosity]
            dLxuvModel = daOutputs[0]  # LXUVStellar
            dLbolModel = daOutputs[1]  # Luminosity
            dLogRatioModel = np.log10(dLxuvModel / dLbolModel)

            f.write("\nModel Predictions (at maximum likelihood):\n")
            f.write("-" * 70 + "\n")
            f.write(f"L_bol (model)            = {dLbolModel:.6e} Lsun\n")
            f.write(f"L_XUV (model)            = {dLxuvModel:.6e} Lsun\n")
            f.write(f"log(L_XUV/L_bol) (model) = {dLogRatioModel:.3f}\n")

            f.write("\nResiduals:\n")
            f.write("-" * 70 + "\n")
            f.write(f"ΔL_bol            = {(dLbolModel - dLBolObs)/dLBolStd:.3f} σ\n")
            f.write(f"Δlog(L_XUV/L_bol) = {(dLogRatioModel - dLogLxuvLbolObs)/dLogLxuvLbolStd:.3f} σ\n")

        except Exception as e:
            f.write(f"\nCould not calculate model predictions: {e}\n")

        f.write("\nParameter Bounds:\n")
        f.write("-" * 70 + "\n")
        for i, sName in enumerate(listParamNames):
            f.write(f"{sName:15s}: [{daParamBounds[i,0]:.6e}, {daParamBounds[i,1]:.6e}]\n")

    print(f"✓ Results saved: {sResultFile}")
    return True

# ===========================================================================
# Main Execution
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GJ 1132 Ribas XUV Model - Maximum A Posteriori (MAP) Estimation")
    print("=" * 70)

    # Find maximum a posteriori parameters
    daMAPParams, dNegLogPost = daFindMaxLikelihood()

    # Calculate individual components for display
    dNegLogPrior = fdNegLogPrior(daMAPParams)
    dNegLogLike = fdNegLogLikelihood(daMAPParams)

    # Print results
    print("\n" + "=" * 70)
    print("Maximum A Posteriori (MAP) Results")
    print("=" * 70)
    for i, sName in enumerate(listParamNames):
        print(f"{sName:15s} = {daMAPParams[i]:.6e}")
    print(f"\n-ln(Posterior)  = {dNegLogPost:.6e}  (optimized)")
    print(f"-ln(Likelihood) = {dNegLogLike:.6e}")
    print(f"-ln(Prior)      = {dNegLogPrior:.6e}")
    print(f"chi^2           = {2*dNegLogLike:.6e}")

    # Save results
    fbSaveResults(daMAPParams, dNegLogPost)

    # Generate plot
    fbPlotEvolution(daMAPParams, dNegLogPost)

    print("\n" + "=" * 70)
    print("MAP estimation completed successfully!")
    print("=" * 70 + "\n")
