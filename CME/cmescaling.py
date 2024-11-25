import numpy as np
import matplotlib.pyplot as plt
import vplot

# Figure initialization
fig, ax = plt.subplots(nrows = 2, ncols=1, figsize=(6.5, 7))
#fig = plt.figure(figsize=(6.5, 6))

# Contour plot of lambda vs. epsilon
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
contours = ax[0].contour(daGridLambda, daGridEpsilon, daCarrHLost, colors = 'k', levels=[0.3,1,3,10])
#carr_contours = plt.contour(daGridCharLum, daGridFlareLum, daMassCME, colors = 'red', levels=[11.07,12.07,13.07])
ax[0].clabel(contours,inline=True, fontsize=15, fmt="%.1f")
#plt.clabel(carr_contours,inline=True, fontsize=15, fmt="%.2f")

ax[0].tick_params(axis='x',labelsize=14)
ax[0].tick_params(axis='y',labelsize=14)
ax[0].set_xlabel(r"$L_{XUV} / L_{1-8\AA}$ ($\lambda$)",fontsize=18)
ax[0].set_ylabel(r"CME Escape Efficiency ($\epsilon$)",fontsize=18)
ax[0].set_title(r"Mass Lost From Carrington Event [$10^6$ kg]",fontsize=18)

# Mass lost from GJ 1132 b from CMEs
dEpsilon = 0.5
dLambda = 40
dTheta = 2/3
rp = 1.91*rearth
mp = 1.81*mearth
vesc = np.sqrt(2*bigg*mp/rp)
a = 0.0157

daLogLXUVFlare = np.linspace(-12,-3,100)
daMassLostMean = np.log10(dEpsilon*dLambda*rp**2*dTheta/2 * (vproton/vesc) * (10**dLogCarrFluenceMean/dCarrLXLumMean) * a**-2 * mproton * (10**daLogLXUVFlare) * lsun)
daMassLostMin = np.log10(dEpsilon*dLambda*rp**2*dTheta/2 * (vproton/vesc) * (10**(dLogCarrFluenceMean - 3*dLogCarrFluenceSigma)/(dCarrLXLumMean + 3*dLogCarrFluenceSigma)) * a**-2 * mproton * (10**daLogLXUVFlare) * lsun)
daMassLostMax = np.log10(dEpsilon*dLambda*rp**2*dTheta/2 * (vproton/vesc) * (10**(dLogCarrFluenceMean + 3*dLogCarrFluenceSigma)/(dCarrLXLumMean - 3*dCarrLXLumSigma)) * a**-2 * mproton * (10**daLogLXUVFlare) * lsun)

ax[1].tick_params(axis='both',labelsize=14)
ax[1].plot(daLogLXUVFlare,daMassLostMean,'k',label='Mean')
ax[1].plot(daLogLXUVFlare,daMassLostMin,'k',linestyle='dashed',label=r'$3\sigma$')
ax[1].plot(daLogLXUVFlare,daMassLostMax,'k',linestyle='dashed')
ax[1].set_xlabel(r'log$_{10}$($L_{XUV}^{flare}$) [$L_\odot$]',fontsize=18)
ax[1].set_ylabel(r'log$_{10}$(Mass Lost) [kg]',fontsize=18)
ax[1].legend(loc='best')
ax[1].annotate('GJ 1132 b',[-9,14],fontsize=20)


plt.tight_layout()
#plt.show()
plt.savefig('CMEScaling.png',dpi=300)