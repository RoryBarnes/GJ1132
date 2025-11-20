import pathlib
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import vplanet
try:
    import vplot as vpl
except:
    print("Cannot import vplot. Please install vplot.")

#path = pathlib.Path(__file__).parents[0].absolute()
#sys.path.insert(1, str(path.parents[0]))
#from get_args import get_args

out = vplanet.run()

# Plot
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(6.5, 4))

## Upper left: Bolometric Luminosity #
axes[0, 0].plot(out.star.Time, out.star.Luminosity, color='k')
axes[0, 0].set_yscale("log")
axes[0, 0].set_ylim([0.003,0.1])
axes[0, 0].set_xlim([1e7,1e10])
axes[0, 0].set_ylabel(r"Bol. Luminosity (L$_\odot$)")

# center left: XUV luminosity
axes[1, 0].plot(out.star.Time, out.star.LXUVFlare, color=vpl.colors.dark_blue,label='Flares')
axes[1, 0].plot(out.star.Time, out.star.LXUVStellar, color=vpl.colors.red,label='Quiescence')
axes[1, 0].plot(out.star.Time, out.star.LXUVTot, color='k',label='Total')
axes[1, 0].set_yscale("log")
axes[1, 0].set_ylim([1e-6,1e-4])
axes[1, 0].set_xlim([1e7,1e10])
axes[1, 0].set_ylabel(r"XUV Luminosity (L$_\odot$)")
axes[1,0].set_xlabel("Stellar Age (year)")

# Lower left: XUV luminosity by flares / XUV quiescent luminosity
axes[0,1].plot(out.star.Time, out.star.LXUVFlare/out.star.LXUVStellar,
    color='k') 
axes[0,1].set_xscale("log")
axes[0,1].set_ylim([0.1,10])
axes[0, 1].set_xlim([1e7,1e10])
axes[0,1].set_ylabel(r"L$_{XUV}^{Flare}$/L$_{XUV}^{Quiescence}$")

# Upper right: XUV quiescent luminosity
axes[1, 1].plot(out.star.Time, out.star.LXUVFlare/out.star.Luminosity,color=vpl.colors.dark_blue,label='Flares')
axes[1, 1].plot(out.star.Time, out.star.LXUVStellar/out.star.Luminosity,color=vpl.colors.red,label='Quiescence')
axes[1, 1].plot(out.star.Time, out.star.LXUVTot/out.star.Luminosity,color='k',label='Total')
axes[1, 1].set_yscale("log")
axes[1, 1].set_xscale("log")
axes[1,1].set_ylabel(r"L$_{XUV}$/L$_{bol}$")
axes[1, 1].legend(loc="lower left", ncol=1, fontsize=7)
axes[1,1].set_xlabel("Stellar Age (year)")
axes[1,1].set_xlim([1e7,1e10])


# Saving figure
#ext = get_args().ext
fig.savefig("LXUV.png", bbox_inches="tight", dpi=300)
