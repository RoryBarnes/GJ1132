import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import vplot

import vplanet

# Path hacks
path = pathlib.Path(__file__).parents[0].absolute()
sys.path.insert(1, str(path.parents[0]))

# Run vplanet
out = vplanet.run()
time = out.star.Time / 1e3

#################### 4th order DistOrb ####################

fig = plt.figure(figsize=(6.5, 5))
plt.subplot(2, 2, 1)
plt.plot(time, out.b.Obliquity, color='k')
plt.xlabel("Time (kyr)")
plt.ylabel(r"Obliquity ($^\circ$)")
plt.yscale('log')

plt.subplot(2, 2, 2)
plt.plot(time, out.b.SurfEnFluxTotal, color='k')
plt.xlabel("Time (kyr)")
plt.ylabel(r"Surf. Energy Flux (W/m$^2$)")
#plt.ylim(0.95,1.65)
plt.yscale('log')

plt.subplot(2, 2, 3)
plt.plot(time, out.b.CassiniOne, color='k')
plt.xlabel("Time (kyr)")
plt.ylabel("$\sin{\Psi}$")

plt.subplot(2, 2, 4)
plt.plot(time, out.b.CassiniTwo, color='k')
plt.xlabel("Time (kyr)")
plt.ylabel("$\cos{\Psi}$")

# Save the figure
fig.tight_layout()
fig.savefig("ObliquityExample.png", dpi=300)
