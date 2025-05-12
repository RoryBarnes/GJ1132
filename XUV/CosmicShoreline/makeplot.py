import os
import pathlib
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import vplot as vpl
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import vplanet

path = pathlib.Path(__file__).parents[0].absolute()
sys.path.insert(1, str(path.parents[0]))

output = vplanet.run(units = False)

# Plot!
fig = plt.figure(figsize=(3.25, 3))


fxuv_earth = output.log.final.Earth.CumulativeXUVFlux

fxuv = []
fxuv.append(output.log.final.Mercury.CumulativeXUVFlux/fxuv_earth)
fxuv.append(output.log.final.Venus.CumulativeXUVFlux/fxuv_earth)
fxuv.append(output.log.final.Earth.CumulativeXUVFlux/fxuv_earth)
fxuv.append(output.log.final.Mars.CumulativeXUVFlux/fxuv_earth)
fxuv.append(output.log.final.Jupiter.CumulativeXUVFlux/fxuv_earth)
fxuv.append(output.log.final.Saturn.CumulativeXUVFlux/fxuv_earth)
fxuv.append(output.log.final.George.CumulativeXUVFlux/fxuv_earth)
fxuv.append(output.log.final.Neptune.CumulativeXUVFlux/fxuv_earth)

#print(fxuv)

escvel = []
escvel.append(output.log.final.Mercury.EscapeVelocity/1e3)
escvel.append(output.log.final.Venus.EscapeVelocity/1e3)
escvel.append(output.log.final.Earth.EscapeVelocity/1e3)
escvel.append(output.log.final.Mars.EscapeVelocity/1e3)
escvel.append(output.log.final.Jupiter.EscapeVelocity/1e3)
escvel.append(output.log.final.Saturn.EscapeVelocity/1e3)
escvel.append(output.log.final.George.EscapeVelocity/1e3)
escvel.append(output.log.final.Neptune.EscapeVelocity/1e3)

#print(escvel)

shorelinex = []
shorelinex.append(0.2)
shorelinex.append(60)

shoreliney = []
shoreliney.append(1e-6)
shoreliney.append(1e4)

marker=4
font=10


plt.xlabel('Escape Velocity [km/s]')
plt.ylabel('Normalized Cumulative XUV Flux')
plt.plot(shorelinex,shoreliney,color=vpl.colors.pale_blue,linewidth=marker,zorder=-1)
plt.plot(escvel,fxuv,'o',color='k',markersize=marker)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4,1e4)
plt.xlim(1,100)

plt.annotate('Mercury',(1.6,9),fontsize=font)
plt.annotate('Venus',(10.2,2.6),fontsize=font)
plt.annotate('Earth',(12.5,0.5),fontsize=font)
plt.annotate('Mars',(5,0.15),fontsize=font)
plt.annotate('Jupiter',(30,0.05),fontsize=font)
plt.annotate('Saturn',(38,0.0075),fontsize=font)
plt.annotate('Uranus',(23,0.002),fontsize=font)
plt.annotate('Neptune',(26,0.0007),fontsize=font)
plt.annotate('GJ 1132 b',(4,400),fontsize=font)
plt.annotate('Cosmic',(1.5,0.0011),fontsize=font,rotation=45,color=vpl.colors.pale_blue)
plt.annotate('Shoreline',(25,100),fontsize=font,rotation=45,color=vpl.colors.pale_blue)

x=14.81

qlower=360
qupper=1100
yq=400
yqerr = np.array([[qlower], [qupper]])
plt.plot(x,yq,'o',color='gray', markersize=marker)
plt.errorbar([x],[yq],yerr=[[qlower],[qupper]],capsize=3, capthick=1, elinewidth=1, fmt='none',ecolor='gray')

x=15.25

flower=110
fupper=1600

flower=1190
yf=1300
yferr=np.array([[flower], [fupper]])
plt.plot(x,yf,'o',color='black', markersize=marker)
plt.errorbar([x],[yf],yerr=[[flower],[fupper]],capsize=3, capthick=1, elinewidth=1, fmt='none',ecolor='k')

#plt.axhline(y=yf+fupper, color='red', linestyle='--', alpha=0.3, label=f'Upper bound: {yf+fupper}')
#plt.axhline(y=yf-flower, color='blue', linestyle='--', alpha=0.3, label=f'Lower bound: {yf-flower}')

# Save figure
fig.savefig(path / f"CosmicShoreline.png", bbox_inches="tight", dpi=300)
#plt.show()
