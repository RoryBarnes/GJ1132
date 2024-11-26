import pathlib
import sys
from math import exp, expm1, log, log10
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import vplanet
try:
    import vplot as vpl
except:
    print("Cannot import vplot. Please install vplot.")
import random

random.seed(100)
out = vplanet.get_output()
time = out.star.Time/1e9
l = out.star.Luminosity
lstellar = out.star.LXUVStellar/l
lflare = out.star.LXUVFlare/l
ltot = out.star.LXUVTot/l
flare1 = out.star.FlareFreq1
flare2 = out.star.FlareFreq2
flare3 = out.star.FlareFreq3
flare4 = out.star.FlareFreq4
flare = (flare1 + flare2 + flare3 + flare4)*365.25

upflare=[0 for i in range(len(flare))]
lowflare = [0 for i in range(len(flare))]
for i in range(len(flare)):
    upflare[i] = flare[i].value + ((6)/(i+1)**0.15)
    lowflare[i] = flare[i].value - ((5)/(i+1)**0.175)


sun = vplanet.get_output('Sun')
time = sun.star.Time/1e9
lsun = sun.star.Luminosity
lsunstellar = sun.star.LXUVStellar/lsun
lsunflare = sun.star.LXUVFlare/lsun
lsuntot = sun.star.LXUVTot/lsun
flare1 = sun.star.FlareFreq1
flare2 = sun.star.FlareFreq2
flare3 = sun.star.FlareFreq3
flare4 = sun.star.FlareFreq4
sunflare = (flare1 + flare2 + flare3 + flare4)*365.25


nfig=1
plt.figure(nfig,figsize= (6.5,4))
plt.subplots_adjust(hspace=.5)
### XUV Luminosity
plt.subplot(121)
plt.plot([1e-3,8],[0.00112,0.00112],color='black',linestyle='dotted')
plt.text(0.7,1.5e-3,r'$\beta_{XUV}$',rotation=285)
plt.plot([2,2],[1e-4,2e-3],color='black',linestyle='dotted')
plt.text(2.1,1.6e-3,r'$t_{sat}$',rotation=270)
plt.plot([1.21,8],[2e-3,2.15e-4],color='black',linestyle='dotted')
plt.text(4,1.175e-3,r'$f_{sat}$')

plt.plot(time,ltot,color=vpl.colors.red,label="Total")
plt.plot(time,lstellar,color=vpl.colors.orange,label="Quiescent")
plt.plot(time,lflare,color=vpl.colors.pale_blue,label="Flares")
#plt.plot(time,lsuntot,color=vpl.colors.red,linestyle='dashed')
#plt.plot(time,lsunstellar,color=vpl.colors.orange,linestyle='dashed')
#plt.plot(time,lsunflare,color=vpl.colors.pale_blue,linestyle='dashed')

plt.xlabel('Time (Gyr)')
plt.ylabel(r'XUV Luminosity/Total Luminosity')
plt.ylim(1e-4,2e-3)
#plt.xlim(0,1.2)
plt.yscale('log')
plt.xscale('log')
plt.legend()

plt.subplot(122)
plt.plot(time,flare,color='black',label='Best Fit')
plt.plot(time,upflare,color=vpl.colors.purple,label='1$\sigma$ Uncertainty')
plt.plot(time,lowflare,color=vpl.colors.purple)
plt.xlabel('Time (Gyr)')
plt.ylabel(r'N$_{Carrington}$/year')
plt.xscale('log')
plt.ylim(0,60)
plt.legend()

plt.tight_layout()
#plt.savefig('structure.png')
plt.savefig('stellarevol.pdf')

