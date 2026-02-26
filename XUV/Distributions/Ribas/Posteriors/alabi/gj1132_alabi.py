"""
Posterior inference for GJ 1132 XUV luminosity evolution (Ribas model).

Trains a GP surrogate of the VPLanet likelihood via alabi, then samples
posteriors with both emcee and dynesty. Reads MAP parameters from MaxLEV
output and generates a comparison corner plot.

Usage:
    python gj1132_alabi.py

Prerequisites:
    Run MaxLEV first to generate MAP results:
        cd ../../EvolutionPlots/MaximumLikelihood
        maxlev gj1132_ribas.json --workers -1

Output (all in sSaveDir):
    surrogate_model.pkl       Trained GP surrogate
    emcee_samples.npz         Emcee posterior samples
    dynesty_samples.npz       Dynesty posterior samples
    sampler_comparison.pdf    Corner plot comparing both samplers
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import re
import multiprocessing
import numpy as np
from functools import partial
from pathlib import Path

import matplotlib
import vplot
import matplotlib.pyplot as plt
import corner

import alabi
from alabi import SurrogateModel
from alabi import utility as ut
from sklearn import preprocessing

import vplanet_inference as vpi
import astropy.units as u

# ===========================================================================
# Configuration
# ===========================================================================

# Canonical parameter ordering: [dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta]
listParamLabels = [
    r"$m_{\star}$ [M$_{\odot}$]",
    r"$f_{sat}$",
    r"$t_{sat}$ [Gyr]",
    r"Age [Gyr]",
    r"$\beta_{XUV}$",
]

listBounds = [
    (0.17, 0.22),       # dMass [Msun]
    (-4.0, -2.15),      # dSatXUVFrac (log10)
    (0.1, 5.0),         # dSatXUVTime [Gyr]
    (1.0, 13.0),        # dStopTime (age) [Gyr]
    (0.4, 2.1),         # dXUVBeta
]

# (mean, std) for symmetric Gaussian; (mean, std_pos, std_neg) for asymmetric; (None, None) for uniform
listPriorData = [
    (0.1945, 0.0048, 0.0046),  # mass - asymmetric Gaussian
    (-2.92, 0.26),              # log(fsat) - symmetric Gaussian
    (None, None),               # tsat - uniform (no prior)
    (5.75, 1.38),               # age - symmetric Gaussian
    (1.18, 0.31),               # beta - symmetric Gaussian
]

# Observational constraints: [[mean, std], ...]
daLikeData = np.array([
    [4.38e-3, 3.4e-4],  # Lbol [Lsun]
    [-4.26, 0.15],       # log(Lxuv/Lbol)
])

iNumDimensions = len(listBounds)
iNumTraining = 100 * iNumDimensions
iNumTest = 100 * iNumDimensions
iActiveIterations = 500
iNumCores = max(1, multiprocessing.cpu_count() - 1)

sInpath = os.path.dirname(os.path.abspath(__file__))
sSaveDir = "output/"
sMaxLevResultsPath = str(
    Path(sInpath).parent.parent.parent.parent / "EvolutionPlots"
    / "MaximumLikelihood" / "maxlike_results.txt"
)

# GP hyperparameter configuration
dictGPConfig = {
    "kernel": "ExpSquaredKernel",
    "fit_amp": True,
    "fit_mean": False,
    "fit_white_noise": False,
    "white_noise": -12,
    "uniform_scales": False,
    "theta_scaler": preprocessing.StandardScaler(),
    "y_scaler": preprocessing.StandardScaler(),
    "hyperopt_method": "cv",
    "gp_opt_method": "l-bfgs-b",
    "gp_scale_rng": [-2, 6],
    "gp_amp_rng": [-1, 1],
    "cv_folds": 20,
    "cv_n_candidates": 400,
    "cv_stage2_candidates": 200,
    "cv_stage2_width": 0.3,
    "cv_stage3_candidates": 100,
    "cv_stage3_width": 0.1,
    "cv_weighted_factor": 1.0,
    "multi_proc": True,
}

dictMLOptConfig = {
    "hyperopt_method": "ml",
    "optimizer_kwargs": {
        "maxiter": 100,
        "xatol": 1e-4,
        "fatol": 1e-3,
        "adaptive": True,
    },
}

dictActiveLearningConfig = {
    "algorithm": "bape",
    "gp_opt_freq": 50,
    "obj_opt_method": "nelder-mead",
    "use_grad_opt": False,
    "nopt": 1,
    "optimizer_kwargs": {
        "max_iter": 200,
        "xatol": 1e-3,
        "fatol": 1e-2,
        "adaptive": True,
    },
}

dictEmceeConfig = {
    "nwalkers": 50 * iNumDimensions,
    "nsteps": int(1e5),
    "min_ess": int(1e4),
}

dictDynestyConfig = {
    "sampler_kwargs": {
        "bound": "multi",
        "nlive": 10 * iNumDimensions,
        "sample": "rslice",
        "bootstrap": 0,
    },
    "run_kwargs": {
        "dlogz_init": 1.0,
        "print_progress": True,
    },
    "min_ess": int(1e4),
}

# Plotting parameters
dTickFontsize = 14
dLabelFontsize = 20
dLegendFontsize = 20
dFigsize = 12

# ===========================================================================
# VPLanet Model
# ===========================================================================

vpm = None  # Module-level; initialized in main


def fnInitVplanetModel():
    """Initialize the VPLanet forward model for GJ 1132."""
    global vpm
    dictInparams = {
        "star.dMass": u.Msun,
        "star.dSatXUVFrac": u.dex(u.dimensionless_unscaled),
        "star.dSatXUVTime": u.Gyr,
        "vpl.dStopTime": u.Gyr,
        "star.dXUVBeta": u.dimensionless_unscaled,
    }
    dictOutparams = {
        "final.star.Luminosity": u.Lsun,
        "final.star.LXUVStellar": u.Lsun,
    }
    vpm = vpi.VplanetModel(
        dictInparams, inpath=sInpath, outparams=dictOutparams, verbose=False
    )

# ===========================================================================
# Likelihood, Prior, and Posterior
# ===========================================================================


def lnlike(daTheta):
    """Alias for fdLogLikelihood so pickle can resolve the old surrogate model."""
    return fdLogLikelihood(daTheta)


def fdLogLikelihood(daTheta):
    """Compute log-likelihood by running VPLanet and comparing to observations."""
    try:
        daOutput = vpm.run_model(daTheta, remove=True)
    except Exception:
        return -1e10
    dLbol = daOutput[1]
    dLxuv = daOutput[0]
    if not (np.isfinite(dLbol) and np.isfinite(dLxuv) and dLbol > 0 and dLxuv > 0):
        return -1e10
    daModel = np.array([dLbol, np.log10(dLxuv / dLbol)])
    return -0.5 * np.sum(((daModel - daLikeData[:, 0]) / daLikeData[:, 1]) ** 2)


def fdLogPrior(daTheta):
    """Evaluate log-prior with Gaussian (symmetric/asymmetric) and uniform priors."""
    daTheta = np.atleast_1d(daTheta).flatten()
    for i, (dLower, dUpper) in enumerate(listBounds):
        if not (dLower <= float(daTheta[i]) <= dUpper):
            return -np.inf
    dLogPrior = 0.0
    for i, tPrior in enumerate(listPriorData):
        if tPrior[0] is None:
            continue
        dValue = float(daTheta[i])
        if len(tPrior) == 2:
            dLogPrior += -0.5 * ((dValue - tPrior[0]) / tPrior[1]) ** 2
        elif len(tPrior) == 3:
            dStd = tPrior[1] if dValue >= tPrior[0] else tPrior[2]
            dLogPrior += -0.5 * ((dValue - tPrior[0]) / dStd) ** 2
    return dLogPrior


def ffnSafeSurrogate(fnSurrogateFunction, dFloorValue=-1e4):
    """Clamp GP surrogate output to prevent extrapolation crashes in dynesty.

    Dynesty explores the full prior volume, which can push the GP into
    untrained regions where it returns non-finite or wildly positive values.
    """
    def fnWrapper(daTheta):
        dValue = fnSurrogateFunction(daTheta)
        if not np.isfinite(dValue) or dValue > 0:
            return dFloorValue
        return max(dValue, dFloorValue)
    return fnWrapper


def ffnVarianceAwareSurrogate(fnSurrogateWithVariance, dFloorValue=-1e4,
                               dMaxVariance=100.0):
    """Clamp GP surrogate output and reject high-variance extrapolations.

    Unlike ffnSafeSurrogate, this wrapper also checks the GP predictive
    variance.  Points where the GP is highly uncertain (variance >
    dMaxVariance) are floored, preventing walkers from drifting into
    untrained regions where the GP may extrapolate wildly.
    """
    def fnWrapper(daTheta):
        dMean, dVariance = fnSurrogateWithVariance(daTheta)
        if not np.isfinite(dMean) or dMean > 0 or dVariance > dMaxVariance:
            return dFloorValue
        return max(dMean, dFloorValue)
    return fnWrapper


def fdaGenerateWalkerPositions(daMapParams, iNumWalkers):
    """Generate emcee walker starting positions centred on MAP parameters.

    Each walker is displaced from daMapParams by a small Gaussian perturbation
    (1% of the bound width per dimension), then clipped to stay within bounds.
    """
    iNumDim = len(daMapParams)
    daBoundWidths = np.array([dUpper - dLower
                              for dLower, dUpper in listBounds])
    daScatter = 0.01 * daBoundWidths
    daPositions = daMapParams + daScatter * np.random.randn(iNumWalkers, iNumDim)
    for i, (dLower, dUpper) in enumerate(listBounds):
        daPositions[:, i] = np.clip(daPositions[:, i], dLower, dUpper)
    return daPositions


# ===========================================================================
# MaxLEV Results Reader
# ===========================================================================

# Maps MaxLEV parameter names (file.param) to canonical index
dictParamIndex = {
    "star.dMass": 0, "dMass": 0,
    "star.dSatXUVFrac": 1, "dSatXUVFrac": 1,
    "star.dSatXUVTime": 2, "dSatXUVTime": 2,
    "vpl.dStopTime": 3, "dStopTime": 3,
    "star.dXUVBeta": 4, "dXUVBeta": 4,
}


def ftReadMaxLevResults(sFilePath):
    """Read MAP parameters and objective value from a MaxLEV results file.

    Returns:
        Tuple of (daParams, dNegLogPosterior) where daParams is in canonical order.
    """
    daParams = np.full(iNumDimensions, np.nan)
    dObjective = np.nan
    with open(sFilePath, "r") as f:
        for sLine in f:
            sStripped = sLine.strip()
            matchParam = re.match(r"^(\S+)\s+=\s+(\S+)", sStripped)
            if matchParam:
                sName = matchParam.group(1)
                if sName in dictParamIndex:
                    daParams[dictParamIndex[sName]] = float(matchParam.group(2))
            matchObjective = re.match(r"^-ln\(Posterior\)\s+=\s+(\S+)", sStripped)
            if matchObjective:
                dObjective = float(matchObjective.group(1))
            matchLikelihood = re.match(r"^-ln\(Likelihood\)\s+=\s+(\S+)", sStripped)
            if matchLikelihood and np.isnan(dObjective):
                dObjective = float(matchLikelihood.group(1))
    bMissing = np.isnan(daParams)
    if np.any(bMissing):
        listMissing = [k for k, v in dictParamIndex.items()
                       if "." in k and bMissing[v]]
        raise ValueError(f"Missing parameters in {sFilePath}: {listMissing}")
    return daParams, dObjective

# ===========================================================================
# Surrogate Model Training
# ===========================================================================


def fiReadActiveIterations(sSaveDir):
    """Read the number of completed active learning iterations from the summary file."""
    sPath = os.path.join(sSaveDir, "surrogate_model.txt")
    try:
        with open(sPath, "r") as f:
            sContent = f.read()
        match = re.search(r"Number of active training samples: (\d+)", sContent)
        return int(match.group(1)) if match else 0
    except FileNotFoundError:
        return 0


def fnTrainSurrogate(sSaveDir):
    """Train a new GP surrogate model from scratch with active learning."""
    print("\nTraining new surrogate model...")
    sm = SurrogateModel(
        lnlike_fn=fdLogLikelihood,
        bounds=listBounds,
        savedir=sSaveDir,
        cache=True,
        verbose=True,
        ncore=iNumCores,
    )
    sm.init_samples(ntrain=iNumTraining, ntest=iNumTest, sampler="lhs")
    sm.init_gp(**dictGPConfig)
    sm.active_train(niter=iActiveIterations, **dictActiveLearningConfig)
    return sm


def fsmLoadOrResumeSurrogate():
    """Load cached surrogate, resume partial training, or train from scratch."""
    sPklPath = os.path.join(sSaveDir, "surrogate_model.pkl")
    if not os.path.exists(sPklPath):
        return fnTrainSurrogate(sSaveDir)

    print(f"\nLoading cached surrogate from {sSaveDir}...")
    sm = alabi.load_model_cache(sSaveDir)
    iCompleted = fiReadActiveIterations(sSaveDir)
    print(f"  Completed {iCompleted}/{iActiveIterations} active learning iterations")

    iRemaining = max(0, iActiveIterations - iCompleted)
    if iRemaining > 0:
        print(f"  Resuming: {iRemaining} iterations remaining")
        sm.opt_gp_kwargs.update(dictMLOptConfig)
        sm.active_train(niter=iRemaining, **dictActiveLearningConfig)
    else:
        print("  Training complete, skipping to sampling")
    return sm

# ===========================================================================
# Posterior Sampling
# ===========================================================================


def fnRunEmcee(sm, daMapParams):
    """Run emcee posterior sampling on the trained surrogate.

    Uses MAP-centred walker initialization, variance-aware GP clamping,
    and differential-evolution proposal moves for efficient sampling of
    the narrow, high-failure-rate likelihood surface.
    """
    import emcee

    sEmceePath = os.path.join(sSaveDir, "emcee_samples.npz")
    if os.path.exists(sEmceePath):
        print(f"\nLoading cached emcee samples from {sEmceePath}...")
        sm.emcee_samples = np.load(sEmceePath)["samples"]
        print(f"  Emcee samples: {sm.emcee_samples.shape}")
        return
    print("\nRunning emcee...")
    fnSurrogateWithVariance = sm.create_cached_surrogate_likelihood(
        iter=iActiveIterations, return_var=True)
    fnSafeLikelihood = ffnVarianceAwareSurrogate(fnSurrogateWithVariance)
    daWalkerPositions = fdaGenerateWalkerPositions(
        daMapParams, dictEmceeConfig["nwalkers"])
    sm.run_emcee(
        like_fn=fnSafeLikelihood,
        prior_fn=fdLogPrior,
        p0=daWalkerPositions,
        nwalkers=dictEmceeConfig["nwalkers"],
        nsteps=dictEmceeConfig["nsteps"],
        min_ess=dictEmceeConfig["min_ess"],
        multi_proc=False,
        samples_file="emcee_samples.npz",
        sampler_kwargs={
            "moves": [(emcee.moves.DEMove(), 0.8),
                      (emcee.moves.DESnookerMove(), 0.2)],
        },
    )
    print(f"  Emcee samples: {sm.emcee_samples.shape}")


def fnRunDynesty(sm):
    """Run dynesty nested sampling on the trained surrogate."""
    sDynestyPath = os.path.join(sSaveDir, "dynesty_samples.npz")
    if os.path.exists(sDynestyPath):
        print(f"\nLoading cached dynesty samples from {sDynestyPath}...")
        sm.dynesty_samples = np.load(sDynestyPath)["samples"]
        print(f"  Dynesty samples: {sm.dynesty_samples.shape}")
        return
    print("\nRunning dynesty...")
    surrogateFunction = sm.create_cached_surrogate_likelihood(iter=iActiveIterations)
    safeSurrogateFunction = ffnSafeSurrogate(surrogateFunction)
    priorTransform = partial(
        ut.prior_transform_normal, bounds=listBounds, data=listPriorData
    )
    sm.run_dynesty(
        like_fn=safeSurrogateFunction,
        prior_transform=priorTransform,
        sampler_kwargs=dictDynestyConfig["sampler_kwargs"],
        run_kwargs=dictDynestyConfig["run_kwargs"],
        min_ess=dictDynestyConfig["min_ess"],
        multi_proc=False,
        samples_file="dynesty_samples.npz",
    )
    print(f"  Dynesty samples: {sm.dynesty_samples.shape}")

# ===========================================================================
# Plotting
# ===========================================================================


def ffigCreateCornerBase(daSamples, sColor):
    """Create a corner plot figure with one sampler's contours."""
    fig = corner.corner(
        daSamples,
        labels=listParamLabels,
        range=listBounds,
        color=sColor,
        bins=30,
        smooth=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 2.0, "color": sColor},
        contour_kwargs={"linewidths": 2.0, "colors": sColor},
        label_kwargs={"fontsize": dLabelFontsize},
        title_kwargs={"fontsize": 12},
        fig=plt.figure(figsize=(dFigsize, dFigsize)),
    )
    return fig


def fnOverlayCornerSamples(fig, daSamples, sColor):
    """Overlay a second sampler's contours onto an existing corner plot."""
    corner.corner(
        daSamples,
        fig=fig,
        range=listBounds,
        color=sColor,
        bins=30,
        smooth=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        hist_kwargs={"density": True, "histtype": "step", "linewidth": 2.0, "color": sColor},
        contour_kwargs={"linewidths": 2.0, "colors": sColor},
    )


def fnAddMaxLikelihoodPoints(fig, daMaxLikeParams):
    """Add ML vertical lines on diagonals and dots on off-diagonal panels."""
    axes = np.array(fig.axes).reshape((iNumDimensions, iNumDimensions))
    for i in range(iNumDimensions):
        ax = axes[i, i]
        dYmin, dYmax = ax.get_ylim()
        ax.plot([daMaxLikeParams[i]] * 2, [0, dYmax],
                color="k", linewidth=2.0, alpha=0.8, zorder=10)
        for j in range(i):
            axes[i, j].plot(
                daMaxLikeParams[j], daMaxLikeParams[i], "ko",
                markersize=8, markeredgewidth=1.5, alpha=0.8, zorder=10,
            )


def fnSetTickFontsize(fig):
    """Set tick label fontsize for all active panels in the corner plot."""
    axes = np.array(fig.axes).reshape((iNumDimensions, iNumDimensions))
    for i in range(iNumDimensions):
        for j in range(iNumDimensions):
            if i >= j:
                axes[i, j].tick_params(axis="both", labelsize=dTickFontsize)


def fnAddPriorsToCorner(fig):
    """Overlay prior distributions (grey dashed) on the diagonal panels."""
    axes = np.array(fig.axes).reshape((iNumDimensions, iNumDimensions))
    for i in range(iNumDimensions):
        ax = axes[i, i]
        dXmin, dXmax = listBounds[i]
        daXrange = np.linspace(dXmin, dXmax, 1000)
        if listPriorData[i][0] is None:
            daYprior = np.ones_like(daXrange) / (dXmax - dXmin)
        elif len(listPriorData[i]) == 2:
            dMean, dStd = listPriorData[i]
            daYprior = fdaGaussian(daXrange, dMean, dStd)
        else:
            daYprior = fdaAsymmetricGaussian(daXrange, *listPriorData[i])
        ax.plot(daXrange, daYprior, color="grey", linewidth=2.0,
                linestyle="--", alpha=0.7, zorder=0)


def fdaGaussian(daX, dMean, dStd):
    """Evaluate a normalized Gaussian PDF."""
    return (1.0 / (dStd * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((daX - dMean) / dStd) ** 2
    )


def fdaAsymmetricGaussian(daX, dMean, dStdPos, dStdNeg):
    """Evaluate a normalized asymmetric Gaussian PDF."""
    daResult = np.empty_like(daX)
    bUpper = daX >= dMean
    daResult[bUpper] = fdaGaussian(daX[bUpper], dMean, dStdPos)
    daResult[~bUpper] = fdaGaussian(daX[~bUpper], dMean, dStdNeg)
    return daResult


def fnAddLegendAndSave(fig, sOutputFile, sColorDynesty, sColorEmcee,
                       bShowMaxLikelihood=True):
    """Add manual legend via figure coordinates, adjust spacing, and save."""
    fig.subplots_adjust(
        hspace=0.05, wspace=0.05,
        left=0.12, right=0.98, bottom=0.10, top=0.95,
    )
    dLegendX = 0.72
    dLegendY = 0.87
    dLineLength = 0.03
    dTextOffset = 0.01
    dYspacing = 0.04
    listEntries = [
        (sColorDynesty, "-", 1.0, None, "Dynesty"),
        (sColorEmcee, "-", 1.0, None, "Emcee"),
        ("grey", "--", 0.7, None, "Prior"),
    ]
    if bShowMaxLikelihood:
        listEntries.append(("k", None, 0.8, "o", "Maximum Likelihood"))
    for iEntry, (sColor, sLinestyle, dAlpha, sMarker, sLabel) in enumerate(listEntries):
        dY = dLegendY - iEntry * dYspacing
        if sMarker:
            fig.lines.append(plt.Line2D(
                [dLegendX + dLineLength / 2], [dY],
                marker=sMarker, color=sColor, markersize=8, linewidth=0,
                markeredgewidth=1.5, alpha=dAlpha, transform=fig.transFigure,
            ))
        else:
            fig.lines.append(plt.Line2D(
                [dLegendX, dLegendX + dLineLength], [dY, dY],
                color=sColor, linewidth=2, linestyle=sLinestyle,
                alpha=dAlpha, transform=fig.transFigure,
            ))
        fig.text(
            dLegendX + dLineLength + dTextOffset, dY, sLabel,
            fontsize=dLegendFontsize, va="center", transform=fig.transFigure,
        )
    fig.savefig(sOutputFile, dpi=300)
    print(f"\n  Saved: {sOutputFile}")

# ===========================================================================
# Summary Statistics
# ===========================================================================


def fnPrintPosteriorComparison(daDynesty, daEmcee):
    """Print comparison statistics between dynesty and emcee posteriors."""
    print("\n" + "=" * 70)
    print("POSTERIOR COMPARISON")
    print("=" * 70)
    for i, sLabel in enumerate(listParamLabels):
        dDynestyMean = np.mean(daDynesty[:, i])
        dDynestyStd = np.std(daDynesty[:, i])
        dEmceeMean = np.mean(daEmcee[:, i])
        dEmceeStd = np.std(daEmcee[:, i])
        dAgreement = abs(dDynestyMean - dEmceeMean) / ((dDynestyStd + dEmceeStd) / 2)
        if dAgreement < 0.5:
            sQuality = "excellent agreement"
        elif dAgreement < 1.0:
            sQuality = "good agreement"
        elif dAgreement < 2.0:
            sQuality = "moderate agreement"
        else:
            sQuality = "poor agreement"
        print(f"\n{sLabel}:")
        print(f"  Dynesty: {dDynestyMean:.6f} +/- {dDynestyStd:.6f}")
        print(f"  Emcee:   {dEmceeMean:.6f} +/- {dEmceeStd:.6f}")
        print(f"  Delta/sigma: {dAgreement:.2f} ({sQuality})")
    print("\n" + "=" * 70)

# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GJ 1132 Ribas XUV Model - Posterior Inference")
    print("=" * 70)

    os.makedirs(sSaveDir, exist_ok=True)
    fnInitVplanetModel()

    daMaxLikeParams = None
    if os.path.exists(sMaxLevResultsPath):
        print(f"\nReading MAP results from {sMaxLevResultsPath}...")
        daMaxLikeParams, dNegLogPost = ftReadMaxLevResults(sMaxLevResultsPath)
        print(f"  MAP parameters: {daMaxLikeParams}")
        print(f"  -ln(Posterior): {dNegLogPost:.6e}")
    else:
        print(f"\nWARNING: MaxLEV results not found at {sMaxLevResultsPath}")
        print("  Skipping MAP point overlay. Run MaxLEV first for the full figure.")

    sm = fsmLoadOrResumeSurrogate()

    if daMaxLikeParams is None:
        raise RuntimeError(
            "MAP parameters required for emcee walker initialization. "
            "Run MaxLEV first: cd ../../EvolutionPlots/MaximumLikelihood && "
            "maxlev gj1132_ribas.json --workers -1"
        )
    fnRunEmcee(sm, daMaxLikeParams)
    fnRunDynesty(sm)

    sColorDynesty = vplot.colors.sPaleBlue
    sColorEmcee = vplot.colors.sOrange
    sOutputFile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        sSaveDir, "sampler_comparison.pdf"
    )

    fig = ffigCreateCornerBase(sm.dynesty_samples, sColorDynesty)
    fnOverlayCornerSamples(fig, sm.emcee_samples, sColorEmcee)
    fnSetTickFontsize(fig)
    if daMaxLikeParams is not None:
        fnAddMaxLikelihoodPoints(fig, daMaxLikeParams)
    fnAddPriorsToCorner(fig)
    fnAddLegendAndSave(
        fig, sOutputFile, sColorDynesty, sColorEmcee,
        bShowMaxLikelihood=(daMaxLikeParams is not None),
    )

    fnPrintPosteriorComparison(sm.dynesty_samples, sm.emcee_samples)

    print("\nDone.")
