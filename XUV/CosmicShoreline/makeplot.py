import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import vplot as vpl

import vplanet

PATH = pathlib.Path(__file__).parents[0].absolute()

D_ESCAPE_VELOCITY = 13.858

D_ENGLE_MEAN = 401.16
D_ENGLE_LOWER = 238.48
D_ENGLE_UPPER = 564.09

D_ENGLE_DAV_MEAN = 478.71
D_ENGLE_DAV_LOWER = 275.49
D_ENGLE_DAV_UPPER = 662.63

D_RIBAS_MEAN = 420.6
D_RIBAS_LOWER = 80.82
D_RIBAS_UPPER = 1104.60

D_RIBAS_DAV_MEAN = 478.31
D_RIBAS_DAV_LOWER = 141.18
D_RIBAS_DAV_UPPER = 1106.13

I_MARKER_SIZE = 6
I_FONT_SIZE = 24
I_TICK_FONT_SIZE = 20

SA_PLANET_NAMES = [
    'Mercury', 'Venus', 'Earth', 'Mars',
    'Jupiter', 'Saturn', 'George', 'Neptune',
]

SA_PLANET_LABELS = [
    ('Mercury', 1.6, 9), ('Venus', 10.2, 2.6),
    ('Earth', 12.5, 0.5), ('Mars', 5, 0.15),
    ('Jupiter', 30, 0.05), ('Saturn', 38, 0.0075),
    ('Uranus', 23, 0.002), ('Neptune', 26, 0.0007),
]


def fnPlotErrorBar(dX, dY, dLower, dUpper, color, dAlpha=1):
    """Plot a data point with asymmetric error bars."""
    daYerr = np.array([[dY - dLower], [dUpper - dY]])
    plt.plot(dX, dY, 'o', color=color, markersize=I_MARKER_SIZE, alpha=dAlpha)
    plt.errorbar([dX], [dY], yerr=daYerr, capsize=5, capthick=3,
                 elinewidth=2, fmt='none', ecolor=color, alpha=dAlpha)


def ftExtractPlanetData(output):
    """Extract normalized XUV flux and escape velocity for all planets."""
    dXuvEarth = output.log.final.Earth.CumulativeXUVFlux
    daXuv = [getattr(output.log.final, s).CumulativeXUVFlux / dXuvEarth
             for s in SA_PLANET_NAMES]
    daEscVel = [getattr(output.log.final, s).EscapeVelocity / 1e3
                for s in SA_PLANET_NAMES]
    return daXuv, daEscVel


def fnPlotSolarSystem(daEscVel, daXuv):
    """Plot solar system planets and the cosmic shoreline."""
    plt.xlabel('Escape Velocity [km/s]', fontsize=I_FONT_SIZE)
    plt.ylabel('Normalized Cumulative XUV Flux', fontsize=I_FONT_SIZE)
    plt.plot([0.2, 60], [1e-6, 1e4], color=vpl.colors.pale_blue,
             linewidth=I_MARKER_SIZE, zorder=-1)
    plt.plot(daEscVel, daXuv, 'o', color='k', markersize=I_MARKER_SIZE)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-4, 1e4)
    plt.xlim(1, 100)
    plt.xticks(fontsize=I_TICK_FONT_SIZE)
    plt.yticks(fontsize=I_TICK_FONT_SIZE)


def fnPlotAnnotations():
    """Add planet labels and cosmic shoreline text."""
    for sName, dX, dY in SA_PLANET_LABELS:
        plt.annotate(sName, (dX, dY), fontsize=I_TICK_FONT_SIZE)
    plt.annotate('GJ 1132 b', (4, 400), fontsize=I_TICK_FONT_SIZE)
    plt.annotate('Cosmic', (1.5, 0.0011), fontsize=I_FONT_SIZE,
                 rotation=45, color=vpl.colors.pale_blue)
    plt.annotate('Shoreline', (25, 100), fontsize=I_FONT_SIZE,
                 rotation=45, color=vpl.colors.pale_blue)


def fnPlotGJ1132ErrorBars():
    """Plot GJ 1132 b data points with error bars."""
    dEsc = D_ESCAPE_VELOCITY
    fnPlotErrorBar(dEsc * 1.02, D_RIBAS_DAV_MEAN,
                   D_RIBAS_DAV_LOWER, D_RIBAS_DAV_UPPER, vpl.colors.orange)
    fnPlotErrorBar(dEsc * 1.06, D_RIBAS_MEAN,
                   D_RIBAS_LOWER, D_RIBAS_UPPER, vpl.colors.orange, 0.5)
    fnPlotErrorBar(dEsc * 0.94, D_ENGLE_MEAN,
                   D_ENGLE_LOWER, D_ENGLE_UPPER, 'grey')
    fnPlotErrorBar(dEsc * 0.98, D_ENGLE_DAV_MEAN,
                   D_ENGLE_DAV_LOWER, D_ENGLE_DAV_UPPER, 'k')


def main():
    """Generate the cosmic shoreline figure."""
    sOutputPath = sys.argv[1] if len(sys.argv) > 1 else str(
        PATH / "CosmicShoreline.png")
    output = vplanet.run(infile=str(PATH / "vpl.in"), units=False)
    daXuv, daEscVel = ftExtractPlanetData(output)

    fig = plt.figure(figsize=(6.5, 6))
    fnPlotSolarSystem(daEscVel, daXuv)
    fnPlotAnnotations()
    fnPlotGJ1132ErrorBars()
    fig.savefig(sOutputPath, bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
