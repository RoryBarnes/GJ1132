### plot_run.py ###
### PD 7/7/21 ###
### PURPOSE: plot output from main.py

import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import sys
from funcs import read_data,compare_to_prem
from PREM import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-runname',dest='runname',default='run1',help='runname is the output file name')
parser.add_argument('-s',dest='savefigs',action='store_true',default=None,help='Boolean to save figures.')
args = parser.parse_args()

filename = args.runname+'.out'
#filename = 'run1'+'.out'
figtype='png'
### Read in data
runname,layers_found,phases_found,ir_comp_layer,ir_phase_layer,glob_variab,i_g_change,labels,data = read_data(filename)

### Define variables from data dict.
keys=data.keys()
for key in keys:
    vars()[key] = np.asarray(data[key])
m_planet = m[0]
R_surface = r[0]
i_cmb = ir_comp_layer[-1]+1
i_icb = -1  #default
if phases_found[-1]=='solid': i_icb = ir_phase_layer[-1]

### Begin Plots ###
### 4x1 Profiles
nfig=1
plt.figure(nfig,figsize= (6,10))
plt.subplots_adjust(hspace=.5)
### M(r)
plt.subplot(411)
plt.plot(r*1e-3,m/m_planet)
plt.xlabel('r');plt.ylabel('Mass/M_planet')
plt.yscale('log'); plt.ylim(1e-4,1)
### g(r)
plt.subplot(412)
plt.plot(r*1e-3,g)
plt.xlabel('r');plt.ylabel('Gravity (m/s^2)')
plt.plot(r[i_g_change]*1e-3,g[i_g_change],'x')
plt.ylim(0,np.max([20,max(g)]))
### P(r)
plt.subplot(413)
plt.plot(r*1e-3,P*1e-9)
plt.xlabel('r');plt.ylabel('Pressure (GPa)')
plt.ylim(0,np.max([400,max(P)*1e-9]))
### rho(r)
plt.subplot(414)
plt.plot(r*1e-3,rho)
plt.plot(r[ir_phase_layer[1]]*1e-3,rho_660_pd,'o',alpha=0.5)
plt.plot(r[ir_phase_layer[1]]*1e-3,rho_660_pv,'o',alpha=0.5)
plt.xlabel('r');plt.ylabel('density (kg/m^3)')
plt.ylim(0,np.max([15e3,max(rho)]))
plt.tight_layout()
if args.savefigs:
    plt.savefig(args.runname+'_'+str(nfig)+'.'+figtype)
    print('saved '+args.runname+'_'+str(nfig)+'.'+figtype)

### 3x1 Thermodynamic Profiles
nfig=nfig+1
plt.figure(nfig,figsize= (6,10))
plt.subplots_adjust(hspace=.5)
### rho(r)
plt.subplot(311)
plt.plot(r*1e-3,rho)
plt.xlabel('r');plt.ylabel('rho')
### T(r)
plt.subplot(312)
plt.plot(r*1e-3,T)
plt.plot(r[i_cmb:i_icb]*1e-3,T_melt_Fe[i_cmb:i_icb])
plt.xlabel('r');plt.ylabel('T')
### Gamma(r)
plt.subplot(313)
plt.plot(r*1e-3,gamma)
plt.xlabel('r');plt.ylabel('gamma')
plt.tight_layout()
if args.savefigs:
    plt.savefig(args.runname+'_'+str(nfig)+'.'+figtype)
    print('saved '+args.runname+'_'+str(nfig)+'.'+figtype)

### Temperature and Melting Profiles
nfig=nfig+1
plt.figure(nfig)
plt.plot(r*1e-3,T)
#plt.plot(r*1e-3,T_melt_Fe)
plt.plot(r[i_cmb:i_icb]*1e-3,T_melt_Fe[i_cmb:i_icb],color='red',label='Fe melting')
plt.xlabel('r'); plt.ylabel('T')
plt.legend()
#plt.ylim(0,6e3)
plt.tight_layout()
if args.savefigs:
    plt.savefig(args.runname+'_'+str(nfig)+'.'+figtype)
    print('saved '+args.runname+'_'+str(nfig)+'.'+figtype)

if runname[6:]=='Earth' or runname[6:]=='earth':
    compare_to_prem(r,m,rho,P,T,gamma,ir_phase_layer,i_cmb,i_icb)
    
#    ### Print out transition layers compared to PREM
#    j=1
#    print("pd to pv  : r=%.0f km z=%.0f km P=%.2f GPa rho(above,below)=(%.0f,%.0f) change rho=%f "%(r[ir_phase_layer[j]]*1e-3,(R_surface-r[ir_phase_layer[j]])*1e-3,P[ir_phase_layer[j]]*1e-9,rho[ir_phase_layer[j]],rho[ir_phase_layer[j]+1],(rho[ir_phase_layer[j]+1]/rho[ir_phase_layer[j]]-1)))
#    print("PREM      : r=%.0f km z=%.0f km P=%.2f GPa rho(above,below)=(%.0f,%d) change rho=%f "\
#              %(R_660*1e-3,(R_surface-R_660)*1e-3,P_660*1e-9,rho_660_pd,\
#              rho_660_pv,(rho_660_pv/rho_660_pd-1)))
#    j=j+1
#    print("pv to ppv : r=%d km z=%d km P=%.2f GPa rho(above,below)=(%d,%d) change rho=%f "%(r[ir_phase_layer[j]]*1e-3,(R_surface-r[ir_phase_layer[j]])*1e-3,P[ir_phase_layer[j]]*1e-9,rho[ir_phase_layer[j]],rho[ir_phase_layer[j]+1],(rho[ir_phase_layer[j]+1]/rho[ir_phase_layer[j]]-1)))
#    j=j+1
#    print("CMB : r=%d km z=%d km P=%.2f GPa rho(above,below)=(%d,%d) change rho=%f "%\
#              (r[i_cmb]*1e-3,(R_surface-r[i_cmb])*1e-3,P[i_cmb]*1e-9,\
#                   rho[i_cmb-1],rho[i_cmb+1],(rho[i_cmb+1]/rho[i_cmb-1]-1)))
#    print("PREM: r=%d km z=%d km P=%.2f GPa rho(above,below)=(%d,%d) change rho=%f "\
#              %(R_cmb*1e-3,(R_surface-R_cmb)*1e-3,P_cmb*1e-9,rho_cmb_man,\
#              rho_cmb_core,(rho_cmb_core/rho_cmb_man-1)))
#    j=j+1
#    print("ICB : r=%d km z=%d km P=%.2f GPa rho(above,below)=(%d,%d) change rho=%f "%\
#              (r[i_icb]*1e-3,(R_surface-r[i_icb])*1e-3,P[i_icb]*1e-9,\
#                   rho[i_icb],rho[i_icb+1],(rho[i_icb+1]/rho[i_icb]-1)))
#    print("PREM: r=%d km z=%d km P=%.2f GPa rho(above,below)=(%d,%d) change rho=%f "\
#              %(R_icb*1e-3,(R_surface-R_icb)*1e-3,P_icb*1e-9,rho_icb_oc,\
#              rho_icb_ic,(rho_icb_ic/rho_icb_oc-1)))
#    #i_1221 = np.argwhere(r==R_icb)[0]
#    i_1221 = np.argmin(np.abs(r-R_icb))
#    print("1221 km: rho=%d T=%.2f gamma=%.4f"%(rho[i_1221],T[i_1221],gamma[i_1221]))
#    print("center : r=%.2f km P=%.2f GPa rho=%d m=%e "%\
#              (r[-1]*1e-3,P[-1]*1e-9,rho[-1],m[-1]))
               

