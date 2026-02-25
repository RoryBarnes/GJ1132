"""
Compare posteriors from dynesty and emcee samplers.

Creates a corner plot with contours from both samplers overlaid.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import corner
import vplot


LABELS = [
    r"$m_{\star}$ [M$_{\odot}$]",
    r"$f_{sat}$",
    r"$t_{sat}$ [Gyr]",
    r"Age [Gyr]",
    r"$\beta_{XUV}$",
]

BOUNDS = [
    (0.17, 0.22),
    (-4.0, -2.15),
    (0.1, 5.0),
    (1.0, 13.0),
    (0.4, 2.1),
]

PRIOR_DATA = [
    (0.1945, 0.0048, 0.0046),
    (-2.92, 0.26),
    (None, None),
    (5.75, 1.38),
    (1.18, 0.31),
]

DA_MAX_LIKELIHOOD = np.array([0.194111, -2.920301, 0.423162, 5.752484, 1.180995])


def fdaLoadSamples(sFilepath):
    """Load samples from an .npz file, raising FileNotFoundError on missing."""
    if not os.path.isfile(sFilepath):
        raise FileNotFoundError(f"Sample file not found: {sFilepath}")
    return np.load(sFilepath)["samples"]


def fnPlotUniformPrior(ax, daXrange, dXmin, dXmax, sColor):
    """Plot a uniform prior distribution on a diagonal panel."""
    daYuniform = np.ones_like(daXrange) / (dXmax - dXmin)
    ax.plot(daXrange, daYuniform, color=sColor,
            linewidth=2.0, linestyle="--", alpha=0.7, zorder=0)


def fnPlotGaussianPrior(ax, daXrange, dMean, dStd, sColor):
    """Plot a symmetric Gaussian prior distribution on a diagonal panel."""
    daYprior = ((1.0 / (dStd * np.sqrt(2 * np.pi)))
                * np.exp(-0.5 * ((daXrange - dMean) / dStd) ** 2))
    ax.plot(daXrange, daYprior, color=sColor,
            linewidth=2.0, linestyle="--", alpha=0.7, zorder=0)


def fnPlotAsymmetricPrior(ax, daXrange, dMean, dStdPos, dStdNeg, sColor):
    """Plot an asymmetric Gaussian prior on a diagonal panel."""
    daYprior = np.zeros_like(daXrange)
    baMaskUpper = daXrange >= dMean
    daYprior[baMaskUpper] = ((1.0 / (dStdPos * np.sqrt(2 * np.pi)))
                             * np.exp(-0.5 * ((daXrange[baMaskUpper] - dMean)
                                              / dStdPos) ** 2))
    baMaskLower = daXrange < dMean
    daYprior[baMaskLower] = ((1.0 / (dStdNeg * np.sqrt(2 * np.pi)))
                             * np.exp(-0.5 * ((daXrange[baMaskLower] - dMean)
                                              / dStdNeg) ** 2))
    ax.plot(daXrange, daYprior, color=sColor,
            linewidth=2.0, linestyle="--", alpha=0.7, zorder=0)


def fnPlotPriors(axes, iNumParams):
    """Overlay prior distributions on the diagonal panels."""
    sColor = "grey"
    for i in range(iNumParams):
        ax = axes[i, i]
        dXmin, dXmax = BOUNDS[i]
        daXrange = np.linspace(dXmin, dXmax, 1000)
        tPrior = PRIOR_DATA[i]

        if tPrior[0] is None:
            fnPlotUniformPrior(ax, daXrange, dXmin, dXmax, sColor)
        elif len(tPrior) == 2:
            fnPlotGaussianPrior(ax, daXrange, tPrior[0], tPrior[1], sColor)
        elif len(tPrior) == 3:
            fnPlotAsymmetricPrior(ax, daXrange, tPrior[0], tPrior[1],
                                  tPrior[2], sColor)


def fnPlotMaxLikelihood(axes, iNumParams):
    """Add maximum-likelihood markers to diagonal and off-diagonal panels."""
    for i in range(iNumParams):
        ax = axes[i, i]
        dYmax = ax.get_ylim()[1]
        ax.plot([DA_MAX_LIKELIHOOD[i]] * 2, [0, dYmax],
                color="k", linewidth=2.0, linestyle="-", alpha=0.8, zorder=10)
        for j in range(i):
            axes[i, j].plot(
                DA_MAX_LIKELIHOOD[j], DA_MAX_LIKELIHOOD[i], "ko",
                markersize=8, markerfacecolor="k",
                markeredgewidth=1.5, alpha=0.8, zorder=10)


def fnAddLegendEntry(fig, dX, dY, dLineLen, dTextOffset, iFontSize,
                     sLabel, sColor, sLinestyle="-", dAlpha=1.0,
                     bMarker=False):
    """Add a single line or marker to the figure legend."""
    if bMarker:
        fig.lines.append(plt.Line2D(
            [dX + dLineLen / 2], [dY],
            marker="o", color=sColor, markersize=8, linewidth=0,
            markerfacecolor=sColor, markeredgewidth=1.5, alpha=dAlpha,
            transform=fig.transFigure))
    else:
        fig.lines.append(plt.Line2D(
            [dX, dX + dLineLen], [dY, dY],
            color=sColor, linewidth=2, linestyle=sLinestyle, alpha=dAlpha,
            transform=fig.transFigure))
    fig.text(dX + dLineLen + dTextOffset, dY, sLabel,
             fontsize=iFontSize, va="center", transform=fig.transFigure)


def fnAddLegend(fig, sColorDynesty, sColorEmcee):
    """Add a manual legend using figure coordinates."""
    iFontSize = 20
    dX, dY = 0.72, 0.87
    dLineLen, dTextOffset, dYspacing = 0.03, 0.01, 0.04

    fnAddLegendEntry(fig, dX, dY, dLineLen, dTextOffset, iFontSize,
                     "Dynesty", sColorDynesty)
    fnAddLegendEntry(fig, dX, dY - dYspacing, dLineLen, dTextOffset,
                     iFontSize, "Emcee", sColorEmcee)
    fnAddLegendEntry(fig, dX, dY - 2 * dYspacing, dLineLen, dTextOffset,
                     iFontSize, "Prior", "grey", sLinestyle="--", dAlpha=0.7)
    fnAddLegendEntry(fig, dX, dY - 3 * dYspacing, dLineLen, dTextOffset,
                     iFontSize, "Maximum Likelihood", "k", bMarker=True,
                     dAlpha=0.8)


def fnPrintComparison(daDynesty, daEmcee):
    """Print a summary table comparing the two samplers."""
    print("\n" + "=" * 70)
    print("POSTERIOR COMPARISON")
    print("=" * 70)
    for i, sLabel in enumerate(LABELS):
        dDmean, dDstd = np.mean(daDynesty[:, i]), np.std(daDynesty[:, i])
        dEmean, dEstd = np.mean(daEmcee[:, i]), np.std(daEmcee[:, i])
        dAgreement = abs(dDmean - dEmean) / ((dDstd + dEstd) / 2)
        sQuality = ("excellent" if dAgreement < 0.5
                     else "good" if dAgreement < 1.0
                     else "moderate" if dAgreement < 2.0
                     else "poor — investigate!")
        print(f"\n{sLabel}:")
        print(f"  Dynesty: {dDmean:.6f} +/- {dDstd:.6f}")
        print(f"  Emcee:   {dEmean:.6f} +/- {dEstd:.6f}")
        print(f"  Delta/sigma: {dAgreement:.2f} ({sQuality} agreement)")
    print("\n" + "=" * 70)


def ftBuildCornerFigure(daDynesty, daEmcee, sColorDynesty, sColorEmcee):
    """Create the two-sampler corner plot figure, returning (fig, axes)."""
    dictCornerKwargs = dict(
        range=BOUNDS, bins=30, smooth=1.0,
        plot_datapoints=False, plot_density=False, fill_contours=False,
    )
    fig = corner.corner(
        daDynesty, labels=LABELS, color=sColorDynesty,
        hist_kwargs={"density": True, "histtype": "step",
                     "linewidth": 2.0, "color": sColorDynesty},
        contour_kwargs={"linewidths": 2.0, "colors": sColorDynesty},
        label_kwargs={"fontsize": 20}, title_kwargs={"fontsize": 12},
        fig=plt.figure(figsize=(12, 12)),
        **dictCornerKwargs,
    )
    corner.corner(
        daEmcee, fig=fig, color=sColorEmcee,
        hist_kwargs={"density": True, "histtype": "step",
                     "linewidth": 2.0, "color": sColorEmcee},
        contour_kwargs={"linewidths": 2.0, "colors": sColorEmcee},
        **dictCornerKwargs,
    )
    iNumParams = len(LABELS)
    axes = np.array(fig.axes).reshape((iNumParams, iNumParams))
    for i in range(iNumParams):
        for j in range(i + 1):
            axes[i, j].tick_params(axis="both", labelsize=14)
    return fig, axes


def main():
    """Generate the sampler comparison corner plot."""
    sOutputPath = sys.argv[1] if len(sys.argv) > 1 else "sampler_comparison.pdf"

    daDynesty = fdaLoadSamples("output/dynesty_samples.npz")
    daEmcee = fdaLoadSamples("output/emcee_samples.npz")

    sColorDynesty = vplot.colors.sPaleBlue
    sColorEmcee = vplot.colors.sOrange
    iNumParams = len(LABELS)

    print(f"Dynesty samples: {daDynesty.shape}")
    print(f"Emcee samples:   {daEmcee.shape}")

    fig, axes = ftBuildCornerFigure(daDynesty, daEmcee,
                                    sColorDynesty, sColorEmcee)
    fnPlotMaxLikelihood(axes, iNumParams)
    fnPlotPriors(axes, iNumParams)
    fnAddLegend(fig, sColorDynesty, sColorEmcee)

    fig.subplots_adjust(hspace=0.05, wspace=0.05,
                        left=0.12, right=0.98,
                        bottom=0.10, top=0.95)
    fig.savefig(sOutputPath, dpi=300)
    plt.close()
    print(f"\nSaved comparison plot: {sOutputPath}")

    fnPrintComparison(daDynesty, daEmcee)


if __name__ == "__main__":
    main()
