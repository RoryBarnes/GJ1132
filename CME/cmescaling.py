import numpy as np
import matplotlib.pyplot as plt

# Figure initialization
#fig, ax = plt.subplots(1, 2, figsize=(12, 4))
fig = plt.figure(figsize=(6.5, 6))

# Constants
pi = np.arccos(-1)
mproton = 1.6726219e-27
rsun = 6.957e8
msun =  1.988416e30
lsun = 3.846e26
aum = 1.49597870700e11
rearth = 6.3781e6
dLogCarrFluenceMean = 14
dLogCarrFluenceSigma = 1
dCarrLXLumMean = 1.26e21
dCarrLXLumSigma = 1.4e20
openangle = 120*pi/360
f30 = 0.01
vproton = 7.5e7
bigg=6.672e-11
mearth=5.972186e24
vesc = np.sqrt(2*bigg*mearth/rearth)

# Initialize Arrays
iNumLambda = 100
iNumEpsilon = 100
daLambda = np.linspace(1,125,iNumLambda)
daEpsilon = np.linspace(0,1,iNumEpsilon)
daGridLambda, daGridEpsilon = np.meshgrid(daLambda,daEpsilon)

# Math
daCarrHLost = pi*daGridLambda*daGridEpsilon*rearth**2*(vproton/vesc)*10**dLogCarrFluenceMean * mproton
daCarrHLost = daCarrHLost/1e6
#daCarrWaterLost = 0.5*daCarrHLost
#daCarrCO2Lost = daCarrWaterLost

# Plot
contours = plt.contour(daGridLambda, daGridEpsilon, daCarrHLost, colors = 'k', levels=[0.3,1,3,10])
#carr_contours = plt.contour(daGridCharLum, daGridFlareLum, daMassCME, colors = 'red', levels=[11.07,12.07,13.07])
plt.clabel(contours,inline=True, fontsize=15, fmt="%.1f")
#plt.clabel(carr_contours,inline=True, fontsize=15, fmt="%.2f")

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r"$L_{XUV} / L_{1-8\AA}$ ($\lambda$)",fontsize=20)
plt.ylabel(r"CME Escape Efficiency ($\epsilon$)",fontsize=20)
plt.title(r"Mass Lost From Carrington Event [$10^6$ kg]",fontsize=18)

plt.tight_layout()
#plt.show()
plt.savefig('CMEScaling.png',dpi=300)