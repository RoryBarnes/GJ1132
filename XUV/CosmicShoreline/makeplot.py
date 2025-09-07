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

dEscVel = 13.858

dEngleMean = 401.16
dEngleLower = 238.48
dEngleUpper = 564.09

dEngleDavMean = 484.03
dEngleDavLower = 220.44
dEngleDavUpper = 1680.99

dRibasMean = 420.6
dRibasLower = 80.82
dRibasUpper = 1104.60

dRibasDavMean = 492.00
dRibasDavLower = 79.17
dRibasDavUpper = 1609.43

marker=6
font=24
tickfont=20

#########

output = vplanet.run(units = False)

def PlotErrorBar(dX,dY,dLower,dUpper,color,alpha=1):
    yerr = np.array([[dY - dLower], [dUpper - dY]])
    plt.plot(dX,dY,'o',color=color, markersize=marker,alpha=alpha)
    plt.errorbar([dX],[dY],yerr=yerr,capsize=5, capthick=3, elinewidth=2, fmt='none',ecolor=color,alpha=alpha)

# Plot!
fig = plt.figure(figsize=(6.5, 6))


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




plt.xlabel('Escape Velocity [km/s]',fontsize=font)
plt.ylabel('Normalized Cumulative XUV Flux',fontsize=font)
plt.plot(shorelinex,shoreliney,color=vpl.colors.pale_blue,linewidth=marker,zorder=-1)
plt.plot(escvel,fxuv,'o',color='k',markersize=marker)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-4,1e4)
plt.xlim(1,100)
plt.xticks(fontsize=tickfont)
plt.yticks(fontsize=tickfont)

plt.annotate('Mercury',(1.6,9),fontsize=tickfont)
plt.annotate('Venus',(10.2,2.6),fontsize=tickfont)
plt.annotate('Earth',(12.5,0.5),fontsize=tickfont)
plt.annotate('Mars',(5,0.15),fontsize=tickfont)
plt.annotate('Jupiter',(30,0.05),fontsize=tickfont)
plt.annotate('Saturn',(38,0.0075),fontsize=tickfont)
plt.annotate('Uranus',(23,0.002),fontsize=tickfont)
plt.annotate('Neptune',(26,0.0007),fontsize=tickfont)
plt.annotate('GJ 1132 b',(4,400),fontsize=tickfont)
plt.annotate('Cosmic',(1.5,0.0011),fontsize=font,rotation=45,color=vpl.colors.pale_blue)
plt.annotate('Shoreline',(25,100),fontsize=font,rotation=45,color=vpl.colors.pale_blue)

# Engle Only
dX=dEscVel*0.94
PlotErrorBar(dX,dEngleMean,dEngleLower,dEngleUpper,'grey')

# Engle+Davenport
dX=dEscVel*0.98
PlotErrorBar(dX,dEngleDavMean,dEngleDavLower,dEngleDavUpper,'k')

# Ribas+Davenport
dX=dEscVel*1.02
PlotErrorBar(dX,dRibasDavMean,dRibasDavLower,dRibasDavUpper,vpl.colors.orange)

#Ribas onle
dX=dEscVel*1.06
PlotErrorBar(dX,dRibasMean,dRibasLower,dRibasUpper,vpl.colors.orange,0.5)


# Save figure
fig.savefig(path / f"CosmicShoreline.pdf", bbox_inches="tight", dpi=300)
#plt.show()
