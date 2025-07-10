import numpy as np
import matplotlib.pyplot as plt
import vplot

# Figure initialization
#fig, ax = plt.subplots(nrows = 2, ncols=1, figsize=(6.5, 7))
fig = plt.figure(figsize=(6.5, 6))

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

rb=1.191*rearth
mb=1.83*mearth
ab = 0.0157

vesc = np.sqrt(2*bigg*mb/rb)

xuvratio=50
epsilon=0.5

# Earth values for Carrington
# ab=1
# xuvratio=1
# openangle=4*pi
# rb=rearth
# vesc = np.sqrt(2*bigg*mearth/rearth)

mass = xuvratio*epsilon*rb**2*openangle/2 * (vproton/vesc)* (10**dLogCarrFluenceMean/dCarrLXLumMean) * (dCarrLXLumMean) * (1/ab)**2 * mproton

print(mass/1e6)