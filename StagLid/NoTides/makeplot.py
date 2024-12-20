import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#import vplot

import vplanet

# Path hacks
path = pathlib.Path(__file__).parents[0].absolute()
sys.path.insert(1, str(path.parents[0]))

# Tweaks
plt.rcParams.update({"font.size": 16, "legend.fontsize": 16})

# Run vplanet
out = vplanet.run(path / "vpl.in")

# Plots
rows = 3
cols = 2
body = out.b
# Mantle Figure
nfig = 1
fig = plt.figure(nfig, figsize=(6.5, 9))
panel = 1
plt.subplot(rows, cols, panel)
plt.plot(body.Time, body.TMan, color='red', linestyle="-", label="Mantle")
plt.plot(
    body.Time, body.TCore, color='orange', linestyle="-", label="Core"
)
plt.legend(loc="best", frameon=True)
plt.ylabel("Temperature (K)")
plt.xlabel("Time (Gyr)")
plt.ylim(2000, 6000)
#plt.xticks([0, 1, 2, 3, 4])

panel += 1
plt.subplot(rows, cols, panel)
plt.plot(body.Time, body.CrustDepth, color="black", linestyle="-", label="Crust")
plt.plot(body.Time, body.StagLidDepth, color='red', linestyle="-", label="Stag. Lid")
plt.legend(loc="best", frameon=True)
plt.ylabel("Depth (km)")
plt.xlabel("Time (Gyr)")
plt.ylim(0, 150)
#plt.xticks([0, 1, 2, 3, 4])

panel += 1
plt.subplot(rows, cols, panel)
plt.plot(body.Time, body.WaterOutgasRate/1e12, label=r"$H_2O$")
plt.plot(body.Time, body.CO2OutgasRate/1e12, 'k', label=r"$CO_2$")
plt.legend(loc="best", frameon=True)
plt.ylabel("Outgassing Rate (pg/s)")
plt.xlabel("Time (Gyr)")
#plt.xticks([0, 1, 2, 3, 4])

panel += 1
plt.subplot(rows, cols, panel)
plt.semilogy(body.Time, body.SurfaceCO2Mass, color='blue',label="Atm.")
plt.semilogy(body.Time, body.CrustCO2Mass, color='k',label="Crust")
plt.semilogy(body.Time, body.ManCO2Mass, color='orange', label="Mantle")
plt.legend(loc="best", frameon=True)
plt.ylabel(r"CO$_2$ Mass (bars)")
plt.xlabel("Time (Gyr)")
#plt.xticks([0, 1, 2, 3, 4])

panel += 1
plt.subplot(rows, cols, panel)
plt.semilogy(body.Time, body.SurfWaterMass, color='blue',label="Atm.")
plt.semilogy(body.Time, body.CrustWaterMass, color='k',label="Crust")
plt.semilogy(body.Time, body.ManWaterMass, color='orange', label="Mantle")
plt.ylabel(r"Water Mass (TO)")
plt.xlabel("Time (Gyr)")
plt.legend(loc="best", frameon=True)
plt.ylim(1e-3, 10)
#plt.xticks([0, 1, 2, 3, 4])

panel += 1
plt.subplot(rows, cols, panel)
plt.plot(body.Time, body.MagMom, 'k')
plt.ylabel(r"Mag. Mom. ($\mathcal{M}_\oplus$)")
plt.xlabel("Time (Gyr)")
#plt.xticks([0, 1, 2, 3, 4])

# Save the figure
#ext = get_args().ext
plt.tight_layout()
fig.savefig(path / "NoTidesOutgassing.png",)

