"""Plot GP surrogate convergence on held-out test points.

Uses alabi's internal test_scaled_mse metric, which normalizes the MSE on
held-out test points by the variance of the training targets:

    test_scaled_mse = mean((y_true - y_gp)^2) / var(y_train)

This is equivalent to 1 - R^2: a value of 0 means perfect prediction, while
a value >= 1 means the GP predicts no better than the training mean.

The plot shows test and training normalized RMSE (100 * sqrt(scaled_mse)) vs
active learning iteration, with a horizontal line at the 1% threshold.

Usage:
    python plotConvergenceTest.py [output_path.pdf]
"""

import os
import pickle
import sys

import numpy as np
import vplot
import matplotlib.pyplot as plt


# -- Configuration -----------------------------------------------------------

sSaveDir = "output/"
sDefaultOutput = os.path.join(sSaveDir, "convergence_test.pdf")
dThreshold = 1.0


# -- Surrogate loader --------------------------------------------------------


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


# -- Metric extraction -------------------------------------------------------


def fdaNormalizedRMSE(daScaledMSE):
    """Convert scaled MSE to normalized RMSE as a percentage."""
    return 100.0 * np.sqrt(daScaledMSE)


# -- Plotting -----------------------------------------------------------------


def fnPlotConvergence(ax, daIterations, daTestPct, daTrainPct):
    """Plot test and training normalized RMSE vs iteration."""
    ax.plot(
        daIterations, daTestPct,
        color=vplot.colors.sOrange, linewidth=1.5, label="Test"
    )
    ax.plot(
        daIterations, daTrainPct,
        color=vplot.colors.sDarkBlue, linewidth=1.5, label="Training"
    )
    dAllMin = min(np.min(daTrainPct), np.min(daTestPct))
    dAllMax = max(np.max(daTrainPct), np.max(daTestPct))
    dMargin = 0.1 * (dAllMax - dAllMin)
    ax.set_ylim(max(0, dAllMin - dMargin), dAllMax + dMargin)
    ax.set_xlabel("Active learning iteration")
    ax.set_ylabel("Normalized RMSE (\%)")
    ax.legend(loc="upper right", fontsize=10)


def fnPrintSummary(daIterations, daTestPct, daTrainPct):
    """Print convergence summary to stdout."""
    iMinIdx = np.argmin(daTestPct)
    dMinTest = daTestPct[iMinIdx]
    iMinIter = int(daIterations[iMinIdx])

    print("\n" + "=" * 60)
    print("GP SURROGATE CONVERGENCE TEST SUMMARY")
    print("=" * 60)
    print(f"  Active learning iterations: {len(daIterations)}")
    print(f"  Final test NRMSE:           {daTestPct[-1]:.2f}%")
    print(f"  Final training NRMSE:       {daTrainPct[-1]:.2f}%")
    print(f"  Best test NRMSE:            {dMinTest:.2f}%"
          f"  (iteration {iMinIter})")

    baBelow = daTestPct < dThreshold
    if np.any(baBelow):
        iFirst = int(daIterations[np.argmax(baBelow)])
        print(f"  First below {dThreshold:.0f}%:            "
              f"iteration {iFirst}")
    else:
        print(f"  Never dropped below {dThreshold:.0f}%")

    dRatio = daTestPct[-1] / daTrainPct[-1]
    print(f"\n  Test/Train ratio at end:    {dRatio:.1f}x")
    if dRatio > 10.0:
        print("  ** OVERFITTING: test error >> training error **")
        print(f"  ** Best test NRMSE was at iteration {iMinIter};"
              f" subsequent iterations degraded generalization **")
    print("=" * 60)


# -- Main ---------------------------------------------------------------------


def fnMain():
    """Load stored convergence metrics, plot, and print summary."""
    sOutputPath = sys.argv[1] if len(sys.argv) > 1 else sDefaultOutput

    sm = fsmLoadSurrogate(sSaveDir)
    dictResults = sm.training_results

    daIterations = np.array(dictResults["iteration"])
    daTestPct = fdaNormalizedRMSE(np.array(dictResults["test_scaled_mse"]))
    daTrainPct = fdaNormalizedRMSE(
        np.array(dictResults["training_scaled_mse"])
    )

    fnPrintSummary(daIterations, daTestPct, daTrainPct)

    fig, ax = plt.subplots(figsize=(8, 5))
    fnPlotConvergence(ax, daIterations, daTestPct, daTrainPct)
    fig.tight_layout()
    fig.savefig(sOutputPath, dpi=300)
    print(f"\n  Saved: {sOutputPath}")


if __name__ == "__main__":
    fnMain()
