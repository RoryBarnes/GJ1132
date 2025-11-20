import vplanet
import vplot as vpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pathlib
import sys

# Path hacks
path = pathlib.Path(__file__).parents[0].absolute()
sys.path.insert(1, str(path.parents[0]))
#from get_args import get_args

fig, ax = plt.subplots(nrows = 4, ncols=2, figsize=(6.5, 9))

def PressureToMass(mp,rp,pressure):
    bigg=6.672e-11
    mass = 4*np.pi*rp**4*pressure/(bigg*mp)
    return mass

tomass = 1.39e21

output = vplanet.run(path / "vpl.in", units=False)

luminosity = output.GJ1132.Luminosity 
lxuv = output.GJ1132.LXUVStellar 
teff = output.GJ1132.Temperature

time = output.b.Time / 1e6
tsurf = output.b.SurfTemp
watermo = output.b.WaterMassMOAtm * tomass
watersolid = output.b.WaterMassSol * tomass
wateratm = output.b.PressWaterAtm 
oxygenmo = output.b.OxygenMassMOAtm 
oxygensol = output.b.OxygenMassSol 
oxygenatm = output.b.PressOxygenAtm
co2mo = output.b.CO2MassMOAtm 
co2sol = output.b.CO2MassSol 
co2atm = output.b.PressCO2Atm
solidradius = output.b.SolidRadius
netflux = output.b.NetFluxAtmo 
radpower = output.b.RadioPower 
tidalpower = output.b.TidalPower

#print(oxygenatm)

mp = output.log.final.b.Mass
rp = output.log.final.b.Radius

wateratm = PressureToMass(mp,rp,wateratm)
oxygenatm = PressureToMass(mp,rp,oxygenatm)
co2atm = PressureToMass(mp,rp,co2atm)

#print(wateratm,watermo,watersolid)
#print(netflux)


ax[0,0].tick_params(axis='both',labelsize=14)
ax[0,0].set_xlabel(r"Time [Myr]",fontsize=18)
ax[0,0].set_ylabel(r"Luminosity [L$_\odot$]",fontsize=18)
ax[0,0].plot(time,luminosity,'k',label='Bolometric')
ax[0,0].plot(time,lxuv*1e3,color=vpl.colors.red,linestyle='dashed',label=r'$10^3 \times$ XUV')
ax[0,0].legend(loc='best')

ax[0,1].tick_params(axis='both',labelsize=14)
ax[0,1].set_xlabel(r"Time [Myr]",fontsize=18)
ax[0,1].set_ylabel(r"Net Flux [W/m$^2$]",fontsize=18)
ax[0,1].set_ylim([0,40])
ax[0,1].plot(time,netflux,'k')

ax[1,0].tick_params(axis='both',labelsize=14)
ax[1,0].set_xlabel(r"Time [Myr]",fontsize=18)
ax[1,0].set_ylabel(r"H$_2$O Mass [kg]",fontsize=18)
ax[1,0].plot(time,wateratm,'k',label="Atmosphere")
ax[1,0].plot(time,watermo,color=vpl.colors.orange,label="Magma Ocean")
ax[1,0].plot(time,watersolid,color=vpl.colors.dark_blue,label="Solid Mantle")
ax[1,0].set_yscale('log')
ax[1,0].legend(loc='best')

ax[1,1].tick_params(axis='both',labelsize=14)
ax[1,1].set_xlabel(r"Time [Myr]",fontsize=18)
ax[1,1].set_ylabel(r"O Mass [kg]",fontsize=18)
ax[1,1].plot(time,oxygenatm,'k',label="Atmosphere")
ax[1,1].plot(time,oxygenmo,color=vpl.colors.orange,label="Magma Ocean")
ax[1,1].plot(time,oxygensol,color=vpl.colors.dark_blue,label="Solid Mantle")
ax[1,1].legend(loc='best')
ax[1,1].set_yscale('log')

ax[2,0].tick_params(axis='both',labelsize=14)
ax[2,0].set_xlabel(r"Time [Myr]",fontsize=18)
ax[2,0].set_ylabel(r"CO$_2$ Mass [kg]",fontsize=18)
ax[2,0].plot(time,co2atm,'k',label="Atmosphere")
ax[2,0].plot(time,co2mo,color=vpl.colors.orange,label="Magma Ocean")
ax[2,0].plot(time,co2sol,color=vpl.colors.dark_blue,label="Solid Mantle")
ax[2,0].set_yscale('log')
ax[2,0].legend(loc='best')

ax[2,1].tick_params(axis='both',labelsize=14)
ax[2,1].set_xlabel(r"Time [Myr]",fontsize=18)
ax[2,1].set_ylabel(r"Surf. Temp. [K]",fontsize=18)
ax[2,1].plot(time,tsurf,'k')

ax[3,0].tick_params(axis='both',labelsize=14)
ax[3,0].set_xlabel(r"Time [Myr]",fontsize=18)
ax[3,0].set_ylabel(r"Power [TW]",fontsize=18)
ax[3,0].plot(time,radpower,'k',label="Radiogenic")
ax[3,0].plot(time,tidalpower,color=vpl.colors.pale_blue,label="Tidal")
ax[3,0].set_ylim([70,75])
ax[3,0].legend(loc='best')

ax[3,1].tick_params(axis='both',labelsize=14)
ax[3,1].set_xlabel(r"Time [Myr]",fontsize=18)
ax[3,1].set_ylabel(r"Solid. Rad. [R$_\oplus$]",fontsize=18)
ax[3,1].plot(time,solidradius,'k')
ax[3,1].set_ylim([1,1.2])

plt.tight_layout()
#plt.show()
plt.savefig('gj1132b.magmoc.png',dpi=300)