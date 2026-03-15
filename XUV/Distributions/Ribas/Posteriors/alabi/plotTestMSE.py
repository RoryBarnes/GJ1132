"""
Plot per-point squared error vs true log-likelihood for the GP test sample.

Checks whether the surrogate is accurate in the high-probability regions
(near log-likelihood = 0) that drive posterior inference.

Usage:
    python plotTestMSE.py
"""

import os
import sys
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
import vplot
import matplotlib.pyplot as plt

import alabi
from gj1132_alabi import (
    fdLogLikelihood, fdLogPrior, fdaPriorTransform,
    fnInitVplanetModel, listParamLabels, listBounds,
    sSaveDir,
)


def fnMain():
    fnInitVplanetModel()

    sm = alabi.load_model_cache(sSaveDir)
    print(f"Loaded surrogate: {sm.ntrain} training, {len(sm._theta_test)} test points")

    daTestTheta = sm._theta_test
    daTestYScaled = sm._y_test

    # GP predictions on the test set (scaled space)
    daGPPredScaled = sm.gp.predict(
        sm._y, daTestTheta, return_cov=False, return_var=False
    )

    # Inverse-transform to real log-likelihood space
    daTestYTrue = sm.y_scaler.inverse_transform(
        daTestYScaled.reshape(-1, 1)
    ).flatten()
    daGPPred = sm.y_scaler.inverse_transform(
        daGPPredScaled.reshape(-1, 1)
    ).flatten()

    daPercentError = 100.0 * np.abs(daTestYTrue - daGPPred) / np.abs(daTestYTrue)

    # Sort by true log-likelihood for cleaner visual
    daOrder = np.argsort(daTestYTrue)
    daTestYTrue = daTestYTrue[daOrder]
    daPercentError = daPercentError[daOrder]
    daGPPred = daGPPred[daOrder]

    fnPlotPercentError(daTestYTrue, daPercentError)
    fnPlotResidual(daTestYTrue, daGPPred)


def fnPlotPercentError(daLogLike, daPercentError):
    """Scatter plot of percent error vs true log-likelihood."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(daLogLike, daPercentError, s=12, alpha=0.6,
               color=vplot.colors.sDarkBlue, edgecolors="none")
    ax.set_xlabel("True log-likelihood", fontsize=14)
    ax.set_ylabel(r"$\left|\,\mathrm{GP} - \mathrm{true}\,\right|"
                   r"\;/\;\left|\,\mathrm{true}\,\right|$ (\%)", fontsize=14)
    ax.set_yscale("log")
    ax.tick_params(labelsize=12)

    # Highlight the high-probability region
    dThreshold = -10.0
    iHighProb = np.sum(daLogLike > dThreshold)
    daHighProbPE = daPercentError[daLogLike > dThreshold]
    if iHighProb > 0:
        dMeanPE = np.mean(daHighProbPE)
        dMaxPE = np.max(daHighProbPE)
        ax.axvline(dThreshold, color="grey", linestyle="--", alpha=0.5)
        ax.text(0.97, 0.97,
                f"ln(L) $>$ {dThreshold}: {iHighProb} pts\n"
                f"mean = {dMeanPE:.2f}\\%\n"
                f"max = {dMaxPE:.2f}\\%",
                transform=ax.transAxes, fontsize=11,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    sOutPath = os.path.join(sSaveDir, "test_mse_vs_loglike.pdf")
    fig.savefig(sOutPath, dpi=200)
    print(f"Saved: {sOutPath}")
    plt.close(fig)


def fnPlotResidual(daLogLike, daGPPred):
    """Residual plot: GP prediction vs true, with 1:1 line."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(daLogLike, daGPPred, s=12, alpha=0.5,
               color=vplot.colors.sOrange, edgecolors="none")
    dMin = min(daLogLike.min(), daGPPred.min())
    dMax = max(daLogLike.max(), daGPPred.max())
    ax.plot([dMin, dMax], [dMin, dMax], "k--", linewidth=1, alpha=0.5)

    ax.set_xlabel("True log-likelihood", fontsize=14)
    ax.set_ylabel("GP predicted log-likelihood", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    sOutPath = os.path.join(sSaveDir, "test_gp_vs_true.pdf")
    fig.savefig(sOutPath, dpi=200)
    print(f"Saved: {sOutPath}")
    plt.close(fig)


if __name__ == "__main__":
    fnMain()
