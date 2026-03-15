"""Rerun all four samplers on a GP truncated at the best active learning iteration.

Reconstructs the GP surrogate using only the first N active learning
points (where test scaled MSE was minimized), then reruns emcee, dynesty,
and ultranest on the truncated surrogate. Compares the truncated posteriors
to the full-training posteriors.

Usage:
    python truncatedConvergence.py [iteration]

If no iteration is given, uses the iteration with minimum test scaled MSE.
"""

import os
import pickle
import sys

import numpy as np
import emcee
import matplotlib
import vplot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.stats import norm

import multiprocessing


# ── Configuration ─────────────────────────────────────────────────────────

sSaveDir = "output/"
sOutputDir = "output/truncated/"

listParamLabels = [
    r"$m_{\star}$ [M$_{\odot}$]",
    r"$f_{sat}$",
    r"$t_{sat}$ [Gyr]",
    r"Age [Gyr]",
    r"$\beta_{XUV}$",
]

listBounds = [
    (0.17, 0.22),
    (-4.0, -2.15),
    (0.1, 5.0),
    (1.0, 13.0),
    (0.4, 2.1),
]

iNumDimensions = len(listBounds)

# Prior data (must match gj1132_alabi.py)
listPriorData = [
    (0.1945, 0.0048, 0.0046),
    (-2.92, 0.26),
    (None, None),
    "empirical",
    (1.18, 0.31),
]

sInpath = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
sAgeSamplesPath = str(
    Path(sInpath).parent.parent.parent / "Engle" / "Age" / "age_samples.txt"
)

dictSamplerColors = {
    "emcee": vplot.colors.sOrange,
    "dynesty": vplot.colors.sPaleBlue,
    "ultranest": vplot.colors.sDarkBlue,
}


# ── Prior functions ───────────────────────────────────────────────────────


def ftBuildEmpiricalPrior(sSamplesPath, tBounds):
    """Load age samples and return (KDE, inverse CDF interpolator)."""
    daAgeGyr = np.loadtxt(sSamplesPath) / 1e9
    daAgeGyr = daAgeGyr[(daAgeGyr >= tBounds[0]) & (daAgeGyr <= tBounds[1])]
    kdeAge = gaussian_kde(daAgeGyr)
    daSorted = np.sort(daAgeGyr)
    daCDF = (np.arange(len(daSorted)) + 0.5) / len(daSorted)
    fnInverseCDF = interp1d(
        daCDF, daSorted, bounds_error=False,
        fill_value=(daSorted[0], daSorted[-1]),
    )
    return kdeAge, fnInverseCDF


kdeAgePrior, fnAgePriorInverseCDF = ftBuildEmpiricalPrior(
    sAgeSamplesPath, listBounds[3]
)


def fdLogDensityEmpirical(dValue):
    """Evaluate log-density of the empirical age prior."""
    dDensity = kdeAgePrior(dValue)[0]
    if dDensity <= 0:
        return -np.inf
    return float(np.log(dDensity))


def fdLogPrior(daTheta):
    """Evaluate log-prior with Gaussian, empirical, and uniform priors."""
    daTheta = np.atleast_1d(daTheta).flatten()
    for i, (dLower, dUpper) in enumerate(listBounds):
        if not (dLower <= float(daTheta[i]) <= dUpper):
            return -np.inf
    dLogPrior = 0.0
    for i, tPrior in enumerate(listPriorData):
        if tPrior == "empirical":
            dLogPrior += fdLogDensityEmpirical(float(daTheta[i]))
            continue
        if tPrior[0] is None:
            continue
        dValue = float(daTheta[i])
        if len(tPrior) == 2:
            dLogPrior += -0.5 * ((dValue - tPrior[0]) / tPrior[1]) ** 2
        elif len(tPrior) == 3:
            dStd = tPrior[1] if dValue >= tPrior[0] else tPrior[2]
            dLogPrior += -0.5 * ((dValue - tPrior[0]) / dStd) ** 2
    return dLogPrior


def fdaPriorTransform(daX):
    """Transform unit hypercube to parameter space."""
    daX = np.asarray(daX, dtype=float)
    daResult = np.zeros(iNumDimensions)
    for i in range(iNumDimensions):
        tPrior = listPriorData[i]
        if tPrior == "empirical":
            daResult[i] = float(fnAgePriorInverseCDF(daX[i]))
        elif tPrior[0] is None:
            dLower, dUpper = listBounds[i]
            daResult[i] = dLower + (dUpper - dLower) * daX[i]
        else:
            daResult[i] = norm.ppf(daX[i], tPrior[0], tPrior[1])
    return daResult


# ── Surrogate loader ──────────────────────────────────────────────────────


def fdLogLikelihood(daTheta):
    """Stub for pickle."""
    return 0.0


def lnlike(daTheta):
    """Stub alias."""
    return 0.0


def fsmLoadSurrogate(sSaveDir):
    """Load surrogate model pickle with stub likelihood."""
    class StubUnpickler(pickle.Unpickler):
        def find_class(self, sModule, sName):
            if sName in ("fdLogLikelihood", "lnlike"):
                return fdLogLikelihood
            return super().find_class(sModule, sName)

    sPklPath = os.path.join(sSaveDir, "surrogate_model.pkl")
    with open(sPklPath, "rb") as f:
        return StubUnpickler(f).load()


# ── Truncated GP construction ────────────────────────────────────────────


def fiFindBestIteration(dictTrainingResults):
    """Return the iteration with minimum test scaled MSE."""
    daTestScaledMSE = np.array(dictTrainingResults["test_scaled_mse"])
    return int(np.argmin(daTestScaledMSE)) + 1


def ffnBuildTruncatedLikelihood(sm, iIteration):
    """Use alabi's cached surrogate with iteration-appropriate hyperparams.

    Temporarily patches the stored hyperparameters so that
    create_cached_surrogate_likelihood uses the iteration-72 values,
    then restores the originals.
    """
    daOriginalHP = sm.training_results["gp_hyperparameters"][-1].copy()
    daIterHP = sm.training_results["gp_hyperparameters"][iIteration - 1]
    sm.training_results["gp_hyperparameters"][-1] = daIterHP

    fnCached = sm.create_cached_surrogate_likelihood(
        iter=iIteration, return_var=False)

    sm.training_results["gp_hyperparameters"][-1] = daOriginalHP

    dFloorValue = -1e4

    def fnSafeLikelihood(daTheta):
        dValue = fnCached(daTheta)
        if not np.isfinite(dValue) or dValue > 0:
            return dFloorValue
        return max(dValue, dFloorValue)

    return fnSafeLikelihood


# ── Sampler runners ───────────────────────────────────────────────────────


def fdaGenerateWalkerPositions(daMapParams, iNumWalkers):
    """Generate emcee walker starting positions near MAP."""
    daBoundWidths = np.array([u - l for l, u in listBounds])
    daScatter = 0.01 * daBoundWidths
    daPositions = daMapParams + daScatter * np.random.randn(
        iNumWalkers, len(daMapParams))
    for i, (dLower, dUpper) in enumerate(listBounds):
        daPositions[:, i] = np.clip(daPositions[:, i], dLower, dUpper)
    return daPositions


def fdLogPosterior(daTheta, fnLikelihood):
    """Evaluate log-posterior = log-prior + log-likelihood."""
    dLogPri = fdLogPrior(daTheta)
    if not np.isfinite(dLogPri):
        return -np.inf
    return dLogPri + fnLikelihood(daTheta)


def fdaRunEmcee(fnLikelihood, daMapParams):
    """Run emcee on the truncated surrogate with autocorrelation stopping."""
    iNumWalkers = 20 * iNumDimensions
    iCheckInterval = 500
    iMaxSteps = 20000
    daPositions = fdaGenerateWalkerPositions(daMapParams, iNumWalkers)
    sampler = emcee.EnsembleSampler(
        iNumWalkers, iNumDimensions, fdLogPosterior,
        args=[fnLikelihood],
        moves=[(emcee.moves.DEMove(), 0.8),
               (emcee.moves.DESnookerMove(), 0.2)],
    )
    for iStep in sampler.sample(daPositions, iterations=iMaxSteps,
                                progress=True):
        if sampler.iteration % iCheckInterval == 0:
            try:
                daTau = sampler.get_autocorr_time(quiet=True)
                bConverged = np.all(daTau * 50 < sampler.iteration)
                if bConverged:
                    print(f"  Converged at step {sampler.iteration}"
                          f" (tau={np.max(daTau):.0f})")
                    break
            except emcee.autocorr.AutocorrError:
                continue
    daTau = sampler.get_autocorr_time(quiet=True)
    iBurnin = int(2 * np.max(daTau))
    iThin = max(int(0.5 * np.min(daTau)), 1)
    return sampler.get_chain(discard=iBurnin, thin=iThin, flat=True)


def fdaRunDynesty(fnLikelihood):
    """Run dynesty on the truncated surrogate."""
    import dynesty
    sampler = dynesty.NestedSampler(
        fnLikelihood, fdaPriorTransform, iNumDimensions,
        bound="multi", nlive=10 * iNumDimensions, sample="rslice",
        bootstrap=0,
    )
    sampler.run_nested(dlogz_init=1.0, print_progress=True)
    results = sampler.results
    return dynesty.utils.resample_equal(
        results.samples, np.exp(results.logwt - results.logz[-1])
    )


def fdaRunUltranest(fnLikelihood):
    """Run ultranest on the truncated surrogate."""
    import ultranest
    sUltranestDir = os.path.join(sOutputDir, "ultranest_truncated")
    listNames = [f"p{i}" for i in range(iNumDimensions)]
    sampler = ultranest.ReactiveNestedSampler(
        listNames, fnLikelihood, fdaPriorTransform,
        log_dir=sUltranestDir, resume="overwrite",
    )
    sampler.run(
        min_num_live_points=100 * iNumDimensions,
        dlogz=0.5, dKL=0.5, frac_remain=0.01, show_status=True,
    )
    return sampler.results["samples"]


# ── Comparison plotting ───────────────────────────────────────────────────


def fnPlotComparison(dictFullSamples, dictTruncSamples, iIteration,
                     sOutputPath):
    """Plot full vs truncated posteriors side by side for each parameter."""
    fig, daAxes = plt.subplots(1, iNumDimensions, figsize=(16, 3.5))

    for iDim in range(iNumDimensions):
        ax = daAxes[iDim]
        for sSampler, sColor in dictSamplerColors.items():
            if sSampler in dictFullSamples:
                ax.hist(dictFullSamples[sSampler][:, iDim], bins=50,
                        density=True, histtype="step", linewidth=1.5,
                        color=sColor, linestyle="-")
            if sSampler in dictTruncSamples:
                ax.hist(dictTruncSamples[sSampler][:, iDim], bins=50,
                        density=True, histtype="step", linewidth=1.5,
                        color=sColor, linestyle="--")
        ax.set_xlabel(listParamLabels[iDim], fontsize=10)
        ax.tick_params(labelsize=9)
        if iDim == 0:
            ax.set_ylabel("Density")

    # Legend: solid = full, dashed = truncated
    from matplotlib.lines import Line2D
    listHandles = []
    for sSampler, sColor in dictSamplerColors.items():
        listHandles.append(
            Line2D([], [], color=sColor, linewidth=1.5, linestyle="-",
                   label=f"{sSampler} (full)"))
        listHandles.append(
            Line2D([], [], color=sColor, linewidth=1.5, linestyle="--",
                   label=f"{sSampler} (iter {iIteration})"))
    fig.legend(handles=listHandles, loc="upper center",
               ncol=len(dictSamplerColors), fontsize=8,
               bbox_to_anchor=(0.5, 1.08))

    fig.tight_layout()
    fig.savefig(sOutputPath, dpi=300, bbox_inches="tight")
    print(f"\n  Saved: {sOutputPath}")


def fnPrintAgreement(dictFullSamples, dictTruncSamples, iIteration):
    """Print full vs truncated agreement per sampler and parameter."""
    print(f"\nFULL vs TRUNCATED (iter {iIteration}) AGREEMENT")
    print("=" * 60)
    for iDim in range(iNumDimensions):
        print(f"\n  {listParamLabels[iDim]}:")
        for sSampler in dictSamplerColors:
            if sSampler not in dictFullSamples:
                continue
            if sSampler not in dictTruncSamples:
                continue
            dMeanFull = np.mean(dictFullSamples[sSampler][:, iDim])
            dStdFull = np.std(dictFullSamples[sSampler][:, iDim])
            dMeanTrunc = np.mean(dictTruncSamples[sSampler][:, iDim])
            dStdTrunc = np.std(dictTruncSamples[sSampler][:, iDim])
            dMeanSigma = (dStdFull + dStdTrunc) / 2.0
            dDeltaSigma = abs(dMeanFull - dMeanTrunc) / dMeanSigma
            dWidthRatio = dStdTrunc / dStdFull
            print(f"    {sSampler:12s}: delta/sigma={dDeltaSigma:.3f}"
                  f"  width_ratio={dWidthRatio:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────


def fnMain():
    """Reconstruct truncated GP and rerun samplers."""
    sm = fsmLoadSurrogate(sSaveDir)

    if len(sys.argv) > 1:
        iIteration = int(sys.argv[1])
    else:
        iIteration = fiFindBestIteration(sm.training_results)

    dScaledMSE = sm.training_results["test_scaled_mse"][iIteration - 1]
    dPercentError = 100.0 * np.sqrt(dScaledMSE)

    print("=" * 60)
    print(f"TRUNCATED GP ANALYSIS at iteration {iIteration}")
    print(f"  Test scaled MSE:  {dScaledMSE:.4e}")
    print(f"  Percent error:    {dPercentError:.2f}%")
    print(f"  Training points:  {sm.ninit_train + iIteration}")
    print("=" * 60)

    os.makedirs(sOutputDir, exist_ok=True)

    fnLikelihood = ffnBuildTruncatedLikelihood(sm, iIteration)

    # Read MAP params from full-run emcee (posterior median as fallback)
    daFullEmcee = np.load(
        os.path.join(sSaveDir, "emcee_samples.npz"))["samples"]
    daMapParams = np.median(daFullEmcee, axis=0)

    # Load full posteriors
    dictFullSamples = {}
    for sSampler in dictSamplerColors:
        sPath = os.path.join(sSaveDir, f"{sSampler}_samples.npz")
        if os.path.exists(sPath):
            dictFullSamples[sSampler] = np.load(sPath)["samples"]

    # Run truncated samplers
    dictTruncSamples = {}

    print("\n--- Emcee (truncated) ---")
    dictTruncSamples["emcee"] = fdaRunEmcee(fnLikelihood, daMapParams)
    print(f"  Samples: {dictTruncSamples['emcee'].shape}")

    print("\n--- Dynesty (truncated) ---")
    dictTruncSamples["dynesty"] = fdaRunDynesty(fnLikelihood)
    print(f"  Samples: {dictTruncSamples['dynesty'].shape}")

    print("\n--- UltraNest (truncated) ---")
    dictTruncSamples["ultranest"] = fdaRunUltranest(fnLikelihood)
    print(f"  Samples: {dictTruncSamples['ultranest'].shape}")

    # Save truncated samples
    for sSampler, daSamples in dictTruncSamples.items():
        sPath = os.path.join(sOutputDir, f"{sSampler}_truncated.npz")
        np.savez(sPath, samples=daSamples)

    fnPrintAgreement(dictFullSamples, dictTruncSamples, iIteration)
    sPlotPath = os.path.join(sOutputDir, "truncated_comparison.pdf")
    fnPlotComparison(dictFullSamples, dictTruncSamples, iIteration, sPlotPath)


if __name__ == "__main__":
    fnMain()
