#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:44:19 2021

@author: carone
"""

import numpy as np
from matplotlib import pyplot as plt 
import vplot


TO        = 1.39e21      # mass of 1 Terr. Ocean [kg]
REARTH = 6.3781e6        # m
MEARTH = 5.972186e24     # kg
BIGG   = 6.67428e-11     # m**3/kg/s**2

MOLWEIGHTCO2=44.01e-3
MOLWEIGHTWATER=18.01528e-3
MOLWEIGHTO2=2.0*15.999e-3
MuH2O = 18
MuO2  = 32

#fig = plt.figure()
fig, axs = plt.subplots(2,2)
#plt.figure(figsize= (6.5,4))


    

#fig.suptitle('GJ1132b magma ocean 100 TO H2O and 20 TO CO2')
    #plt.xlabel("Simulation time [Myrs]") 
    
axs[0,0].set(xlabel='Ttime (Myr)', ylabel='Surface Temperature (K)', xscale='log')
axs[0,1].set(xlabel='Time (Myr)', ylabel='H$_2$O Partial Press. (Bar)', xscale='log', yscale='log')
axs[1,0].set(xlabel='Time (Myr)', ylabel='CO$_2$ Partial Press. (Bar)', xscale='log', yscale='log',ylim=[10,1E5])
axs[1,1].set(xlabel='Time (Myr)', ylabel='O$_2$ Partial Press. (Bar)', xscale='log', yscale='log')
    #plt.ylabel("Temperature [K]") 
#    plt.xscale('log')
    

data = np.loadtxt("GJ1132.GJ1132b.forward")
R_N_Planet = 1.15
M_N_Planet = 1.62

        # write data to arrays
time        = data[:,0]  # time (yr)
TpotCO2        = data[:,1]  # Potential temp magma ocean (K)
TsurfCO2       = data[:,2]  # Surface temp (K)
r_solCO2       = data[:,3]  # solidification radius (R_earth)
M_water_moCO2  = data[:,4] # water mass in magma ocean + atmosphere (TO)
M_water_solCO2 = data[:,5] # water mass in solid mantle (TO)
M_O_moCO2      = data[:,6] # mass of oxygen in magma ocean + atmosphere (kg)
M_O_solCO2     = data[:,7] # mass of oxygen in solid mantle (kg)
Press_H2O_CO2   = data[:,8] # mass pressure water in atmopshere (bar)
Press_O_CO2     = data[:,9] # mass pressure oxygen in atmosphere (bar)
M_H_SpaceCO2   = data[:,10] # partial pressure oxygen in atmosphere (bar)
M_O_SpaceCO2   = data[:,11] # partial pressure oxygen in atmosphere (bar)
Frac_Fe2O3_CO2  = data[:,12] # partial pressure oxygen in atmosphere (bar)
NetFluxAtmoCO2 = data[:,13] # atmospheric net flux (W/m^2)
Frac_H2O_CO2    = data[:,15] # Water fraction in magma ocean
M_CO2_moCO2    = data[:,16] # CO2 mass in magma ocean + atmosphere
M_CO2_solCO2   = data[:,17] # CO2 mass in solid mantle 
Press_CO2_CO2   = data[:,18] # mass pressure CO2 in atmopshere (bar)
Frac_CO2_CO2    = data[:,19] # CO2 fraction in magma ocean


n_time = len(time)
        #Just before the very last step is better for the Earth
i_end  = n_time-2
        #For Trappist, we need the time when R_solid=R_planet
Planet_solid=np.where(r_solCO2 == R_N_Planet)
if len(Planet_solid[0]>0):
            i_end = Planet_solid[0][0]
            t_solid = i_end
    
M_water_atm = np.zeros(n_time)
M_O_atm     = np.zeros(n_time)
M_CO2_atm = np.zeros(n_time)
    
       #Replace with automatically reading from source file as used in print_results
r_p    = R_N_Planet * REARTH
m_p    = M_N_Planet * MEARTH
g      = (BIGG * m_p) / (r_p ** 2)
        #This is the default value check!
r_c    = R_N_Planet * 3.4E6/REARTH
        #0.613035 *REARTH #for TR-1g
        #    r_c    = r_p * 4.3781e6 / REARTH #2000 km thick magma ocean

    


    # find time of solidification, desiccation, and/or entry of habitable zone
for i in range(n_time):
            #This only works, because we output pressures for mass calculations
            #NOT partial pressures
            M_water_atm[i] = Press_H2O_CO2[i] * 1e5 * 4 * np.pi * r_p**2 / g
            M_CO2_atm[i]   = Press_CO2_CO2[i] * 1e5 * 4 * np.pi * r_p**2 / g
            M_O_atm[i]     = Press_O_CO2[i]   * 1e5 * 4 * np.pi * r_p**2 / g

M_water_ini = M_water_moCO2[0]


MOL_ave_tot = (Press_H2O_CO2 + Press_O_CO2 + Press_CO2_CO2)/\
            (Press_H2O_CO2/MOLWEIGHTWATER+Press_CO2_CO2/MOLWEIGHTCO2+Press_O_CO2/MOLWEIGHTO2)
Press_tot = Press_H2O_CO2 + Press_O_CO2 + Press_CO2_CO2

PressPartH2O = Press_H2O_CO2*MOLWEIGHTWATER/MOL_ave_tot
PressPartCO2 = Press_CO2_CO2*MOLWEIGHTCO2/MOL_ave_tot
PressPartO2  = Press_O_CO2*MOLWEIGHTO2/MOL_ave_tot
    

l=100
k=5

axs[0,0].plot(time*10**-6, TpotCO2, label='H$_2$O+CO$_2$',color=vplot.colors.red)
axs[0,1].plot(time*10**-6,PressPartH2O,color=vplot.colors.red)
axs[1,0].plot(time*10**-6,PressPartCO2,color=vplot.colors.red)
axs[1,1].plot(time*10**-6,PressPartO2,color=vplot.colors.red)
axs[1,0].set_ylim(500,1.1e4)
xmin = 1e-4
xmax = 1.5e2
axs[0,0].set_xlim(xmin,xmax)
axs[0,1].set_xlim(xmin,xmax)
axs[1,0].set_xlim(xmin,xmax)
axs[1,1].set_xlim(xmin,xmax)

            #axs[1,1].plot(time*10**-6,MOL_ave_tot, label='$CO_2$ ='+str(l/k)+' TO')
#axs.legend(loc='best')

#axs[0,1].legend()
#axs[1,0].legend()
#axs[1,1].legend()

data = np.loadtxt("../water/GJ1132.GJ1132b.forward")

time        = data[:,0]  # time (yr)
TpotH2O        = data[:,1]  # Potential temp magma ocean (K)
TsurfH2O       = data[:,2]  # Surface temp (K)
r_solH2O       = data[:,3]  # solidification radius (R_earth)
M_water_moH2O  = data[:,4] # water mass in magma ocean + atmosphere (TO)
M_water_solH2O = data[:,5] # water mass in solid mantle (TO)
M_O_moH2O      = data[:,6] # mass of oxygen in magma ocean + atmosphere (kg)
M_O_solH2O     = data[:,7] # mass of oxygen in solid mantle (kg)
Press_H2O_H2O   = data[:,8] # mass pressure water in atmopshere (bar)
Press_O_H2O     = data[:,9] # mass pressure oxygen in atmosphere (bar)
M_H_SpaceH2O   = data[:,10] # partial pressure oxygen in atmosphere (bar)
M_O_SpaceH2O   = data[:,11] # partial pressure oxygen in atmosphere (bar)
Frac_Fe2O3_H2O  = data[:,12] # partial pressure oxygen in atmosphere (bar)
NetFluxAtmoH2O = data[:,13] # atmospheric net flux (W/m^2)
Frac_H2O_H2O    = data[:,15] # Water fraction in magma ocean

n_time = len(time)
        #Just before the very last step is better for the Earth
i_end  = n_time-2
        #For Trappist, we need the time when R_solid=R_planet
Planet_solid=np.where(r_solH2O == R_N_Planet)
if len(Planet_solid[0]>0):
            i_end = Planet_solid[0][0]
            t_solid = i_end
    
M_water_atm = np.zeros(n_time)
M_O_atm     = np.zeros(n_time)
    
       #Replace with automatically reading from source file as used in print_results
r_p    = R_N_Planet * REARTH
m_p    = M_N_Planet * MEARTH
g      = (BIGG * m_p) / (r_p ** 2)
        #This is the default value check!
r_c    = R_N_Planet * 3.4E6/REARTH
        #0.613035 *REARTH #for TR-1g
        #    r_c    = r_p * 4.3781e6 / REARTH #2000 km thick magma ocean

    


    # find time of solidification, desiccation, and/or entry of habitable zone
for i in range(n_time):
            #This only works, because we output pressures for mass calculations
            #NOT partial pressures
            M_water_atm[i] = Press_H2O_H2O[i] * 1e5 * 4 * np.pi * r_p**2 / g
            M_O_atm[i]     = Press_O_H2O[i]   * 1e5 * 4 * np.pi * r_p**2 / g

M_water_ini = M_water_moH2O[0]


MOL_ave_tot = (Press_H2O_H2O + Press_O_H2O)/\
            (Press_H2O_H2O/MOLWEIGHTWATER+Press_O_H2O/MOLWEIGHTO2)
Press_tot = Press_H2O_CO2 + Press_O_CO2 

PressPartH2O = Press_H2O_H2O*MOLWEIGHTWATER/MOL_ave_tot
PressPartO2  = Press_O_H2O*MOLWEIGHTO2/MOL_ave_tot
    

l=100
k=5

axs[0,0].plot(time*10**-6, TpotH2O, label='H$_2$O only',color=vplot.colors.pale_blue)
axs[0,1].plot(time*10**-6,PressPartH2O,color=vplot.colors.pale_blue)
axs[1,1].plot(time*10**-6,PressPartO2,color=vplot.colors.pale_blue)

axs[0,0].legend()


plt.savefig("gj1132magmoc.pdf")

plt.tight_layout()
#plt.show()

