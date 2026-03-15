"""Plot GP surrogate convergence diagnostics and sampler agreement.

Loads the trained surrogate model and the four sampler outputs, then
produces a two-panel figure:
  Top:    test and training scaled MSE vs active learning iteration
  Bottom: 1D marginal histograms from each sampler, overlaid per parameter

Usage:
    python plotConvergence.py [output_path.pdf]
"""

import os
import pickle
import sys

import numpy as np
import matplotlib
import vplot
import matplotlib.pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────────

sSaveDir = "output/"
sDefaultOutput = os.path.join(sSaveDir, "convergence.pdf")
dTargetPercentError = 1.0

listParamLabels = [
    r"$m_{\star}$ [M$_{\odot}$]",
    r"$f_{sat}$",
    r"$t_{sat}$ [Gyr]",
    r"Age [Gyr]",
    r"$\beta_{XUV}$",
]

dictSamplerStyles = {
    "emcee": {"color": vplot.colors.sOrange, "label": "Emcee"},
    "dynesty": {"color": vplot.colors.sPaleBlue, "label": "Dynesty"},
    "multinest": {"color": vplot.colors.sPurple, "label": "MultiNest"},
    "ultranest": {"color": vplot.colors.sDarkBlue, "label": "UltraNest"},
}


# ── Surrogate loader ─────────────────────────────────────────────────────


def fdLogLikelihood(daTheta):
    """Stub so pickle can resolve the likelihood reference."""
    return 0.0


def lnlike(daTheta):
    """Stub alias for fdLogLikelihood."""
    return 0.0


def fsmLoadSurrogate(sSaveDir):
    """Load surrogate model pickle with stub likelihood functions."""
    class StubUnpickler(pickle.Unpickler):
        def find_class(self, sModule, sName):
            if sName in ("fdLogLikelihood", "lnlike"):
                return fdLogLikelihood
            return super().find_class(sModule, sName)

    sPklPath = os.path.join(sSaveDir, "surrogate_model.pkl")
    with open(sPklPath, "rb") as f:
        return StubUnpickler(f).load()


# ── Data loaders ──────────────────────────────────────────────────────────


def fdaLoadSamplerOutput(sSaveDir, sSamplerName):
    """Load posterior samples from a sampler's .npz file."""
    sPath = os.path.join(sSaveDir, f"{sSamplerName}_samples.npz")
    return np.load(sPath)["samples"]


def fdaPercentError(daTestScaledMSE):
    """Convert scaled MSE to root-mean-square percent error."""
    return 100.0 * np.sqrt(daTestScaledMSE)


# ── Plotting ──────────────────────────────────────────────────────────────


def fnPlotActiveLearningConvergence(ax, dictTrainingResults):
    """Plot test and training scaled MSE vs active learning iteration."""
    daIterations = np.array(dictTrainingResults["iteration"])
    daTestScaledMSE = np.array(dictTrainingResults["test_scaled_mse"])
    daTrainScaledMSE = np.array(dictTrainingResults["training_scaled_mse"])

    ax.semilogy(daIterations, daTestScaledMSE, color=vplot.colors.sOrange,
                linewidth=1.5, label="Test")
    ax.semilogy(daIterations, daTrainScaledMSE, color=vplot.colors.sDarkBlue,
                linewidth=1.5, label="Training")

    # Mark the 1% target (scaled MSE = (0.01)^2 = 1e-4)
    dTargetScaledMSE = (dTargetPercentError / 100.0) ** 2
    ax.axhline(dTargetScaledMSE, color="grey", linestyle="--",
               linewidth=1.0, label=f"{dTargetPercentError:.0f}% error")

    ax.set_xlabel("Active learning iteration")
    ax.set_ylabel("Scaled MSE")
    ax.legend(loc="upper right", fontsize=10)


def fnPlotSamplerHistograms(daAxes, dictSamplerSamples):
    """Plot 1D marginal histograms from each sampler, one panel per param."""
    iNumDimensions = len(listParamLabels)
    for iDim in range(iNumDimensions):
        ax = daAxes[iDim]
        for sSampler, dictStyle in dictSamplerStyles.items():
            if sSampler not in dictSamplerSamples:
                continue
            daSamples = dictSamplerSamples[sSampler][:, iDim]
            ax.hist(daSamples, bins=50, density=True, histtype="step",
                    linewidth=1.5, color=dictStyle["color"],
                    label=dictStyle["label"])
        ax.set_xlabel(listParamLabels[iDim], fontsize=10)
        ax.set_ylabel("Density" if iDim == 0 else "")
        ax.tick_params(labelsize=9)
    daAxes[0].legend(loc="upper right", fontsize=8)


def fnPrintConvergenceSummary(dictTrainingResults):
    """Print key convergence metrics to stdout."""
    daTestScaledMSE = np.array(dictTrainingResults["test_scaled_mse"])
    daPercentErr = fdaPercentError(daTestScaledMSE)

    dFinalPercentError = daPercentErr[-1]
    dMinPercentError = np.min(daPercentErr)
    iMinIteration = np.argmin(daPercentErr) + 1

    print("=" * 60)
    print("GP SURROGATE CONVERGENCE SUMMARY")
    print("=" * 60)
    print(f"  Active learning iterations:  {len(daTestScaledMSE)}")
    print(f"  Final percent error:         {dFinalPercentError:.2f}%")
    print(f"  Minimum percent error:       {dMinPercentError:.2f}%"
          f"  (iteration {iMinIteration})")
    print(f"  Final scaled MSE:            {daTestScaledMSE[-1]:.4e}")
    print(f"  Final test MSE:              "
          f"{np.array(dictTrainingResults['test_mse'])[-1]:.4e}")
    print(f"  Final training MSE:          "
          f"{np.array(dictTrainingResults['training_mse'])[-1]:.4e}")

    bBelow = daPercentErr < dTargetPercentError
    if np.any(bBelow):
        iFirst = np.argmax(bBelow) + 1
        print(f"\n  Dropped below {dTargetPercentError:.0f}% at iteration"
              f" {iFirst}")
    else:
        print(f"\n  Never dropped below {dTargetPercentError:.0f}%")
    print("=" * 60)


def fnPrintSamplerAgreement(dictSamplerSamples):
    """Print pairwise mean comparison across samplers."""
    listSamplers = list(dictSamplerSamples.keys())
    iNumDimensions = len(listParamLabels)

    print("\nSAMPLER AGREEMENT (|delta_mean| / mean_sigma)")
    print("-" * 60)
    for iDim in range(iNumDimensions):
        print(f"\n  {listParamLabels[iDim]}:")
        listMeans = []
        listStds = []
        for sSampler in listSamplers:
            dMean = np.mean(dictSamplerSamples[sSampler][:, iDim])
            dStd = np.std(dictSamplerSamples[sSampler][:, iDim])
            listMeans.append(dMean)
            listStds.append(dStd)
            print(f"    {sSampler:12s}: {dMean:.6f} +/- {dStd:.6f}")
        for iA in range(len(listSamplers)):
            for iB in range(iA + 1, len(listSamplers)):
                dDelta = abs(listMeans[iA] - listMeans[iB])
                dMeanSigma = (listStds[iA] + listStds[iB]) / 2.0
                dRatio = dDelta / dMeanSigma if dMeanSigma > 0 else np.inf
                print(f"    {listSamplers[iA]} vs {listSamplers[iB]}:"
                      f" {dRatio:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────


def fnMain():
    """Load data, print summary, and save convergence figure."""
    sOutputPath = sys.argv[1] if len(sys.argv) > 1 else sDefaultOutput

    sm = fsmLoadSurrogate(sSaveDir)
    dictTrainingResults = sm.training_results

    dictSamplerSamples = {}
    for sSampler in dictSamplerStyles:
        try:
            dictSamplerSamples[sSampler] = fdaLoadSamplerOutput(
                sSaveDir, sSampler)
        except FileNotFoundError:
            print(f"  Warning: {sSampler} samples not found, skipping")

    fnPrintConvergenceSummary(dictTrainingResults)
    fnPrintSamplerAgreement(dictSamplerSamples)

    fig = plt.figure(figsize=(10, 8))
    gsTop = fig.add_subplot(2, 1, 1)
    fnPlotActiveLearningConvergence(gsTop, dictTrainingResults)

    iNumDimensions = len(listParamLabels)
    daAxes = [fig.add_subplot(2, iNumDimensions, iNumDimensions + i + 1)
              for i in range(iNumDimensions)]
    fnPlotSamplerHistograms(daAxes, dictSamplerSamples)

    fig.tight_layout()
    fig.savefig(sOutputPath, dpi=300)
    print(f"\n  Saved: {sOutputPath}")


if __name__ == "__main__":
    fnMain()
