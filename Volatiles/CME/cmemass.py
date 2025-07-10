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

iNumFluence = 100
iNumLum = 100
iNumFlare = 100
daLogCarringtonFluence = np.linspace(5,15,iNumFluence)
daLogCarringtonLuminosity = np.linspace(-10,0,iNumLum)
daLogFlareLuminosity = np.linspace(-10, 0, iNumFlare)
daGridCharLum, daGridFlareLum = np.meshgrid(daLogCharacteristicLuminosity,daLogFlareLuminosity)

daMassCME =  2*pi*aum**2 * 10**dLogCarringtonMean * np.power(10,(daGridFlareLum-daGridCharLum)) * (1-np.cos(openangle))* mproton / f30
daMassCME = np.log10(daMassCME)

dLogCarringtonMassMean = np.log10(2*pi*aum**2*10**dLogCarringtonMean * (1-np.cos(openangle)) * mproton / f30)
dLogCarringtonMassMin = np.log10(2*pi*aum**2*10**(dLogCarringtonMean - dLogCarringtonSigma) * (1-np.cos(openangle)) * mproton / f30)
dLogCarringtonMassMax = np.log10(2*pi*aum**2*10**(dLogCarringtonMean + dLogCarringtonSigma) * (1-np.cos(openangle)) * mproton / f30)

print(dLogCarringtonMassMin,dLogCarringtonMassMean,dLogCarringtonMassMax)


contours = plt.contour(daGridCharLum, daGridFlareLum, daMassCME, colors = 'k', levels=[4,8,16,20])
carr_contours = plt.contour(daGridCharLum, daGridFlareLum, daMassCME, colors = 'red', levels=[11.07,12.07,13.07])
plt.clabel(contours,inline=True, fontsize=15, fmt="%.0f")
plt.clabel(carr_contours,inline=True, fontsize=15, fmt="%.2f")

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r"log$_{10}$(Characterisitic Luminosity) [L$_\odot$]",fontsize=20)
plt.ylabel(r"log$_{10}$(Flare XUV Luminosity) [L$_\odot$]",fontsize=20)
plt.title(r"log$_{10}$(CME Mass) [kg]",fontsize=20)

plt.axhline(y = -4.65, color = 'k', linestyle = 'dashed')
plt.axvline(x = -5.7, color = 'k', linestyle = 'dotted') 
plt.axvline(x = -3.7, color = 'k', linestyle = 'dotted') 
plt.annotate('Carrington',[-3.2,-4.5],fontsize=14)
plt.annotate(r'+1$\sigma$',[-3.6,-9.5],fontsize=14,rotation=90)
plt.annotate(r'-1$\sigma$',[-6.05,-9.5],fontsize=14,rotation=90)


plt.tight_layout()
#plt.show()
plt.savefig('CMEMass.png')