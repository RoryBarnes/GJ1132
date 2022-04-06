import numpy as np
from materials import *   #import of the phases properties from file var.py
from constants import *
from PREM import *
import matplotlib.pyplot as plt


def T_melt_Fe_func(rho,P,current_layer,phase,type='Lindemann'):
    if type=='Lindemann':
        if phase=='solid':
            ### need to compute liquid density
            glob_variab_liq,glob_variab_names = get_layer_material('iron',phase='liquid')
            rho_liq = rho-get_density_jump(rho,P,glob_variab_liq)
            #rho_liq = rho+get_density_jump(rho,P,glob_variab_liq)
            #print('rho=%.1f rho_liq=%.1f'%(rho,rho_liq)); exit(0)
        else: rho_liq = rho  #already liquid
        x = rho0_Feliq/rho_liq
        T_melt_Fe = 5400 * np.exp((2 * (gamma0_Feliq/gamma1_Feliq) * (((rho0_Feliq/12166.0)**gamma1_Feliq) - ((x) ** gamma1_Feliq)) + (2/3)*np.log(rho_icb_oc/rho_liq)))
    if type=='Simon-Glatzel':
        T_melt_Fe = T_melt_Fe_SG(P)
        rho_liq = rho  #don't care.
    else:
        print('ERROR!  Melting type unknown!')
        exit(0)
    return(T_melt_Fe,rho_liq)

def T_melt_Fe_SG(P):
    ### This is the Simon-Glatzel law from Morard 2018 supplement.
    P0 = 100.  #[GPa]
    a = 20.   #[GPa]
    #a = 20.
    #c = 3.5  #Morard 2018
    c = 3.0
    #Note: Morard has T0=3500K, which gives T_icb(330GPa)=6039 K
    #T0 = 3362.  #Calibrated to give T_melt=5800K at P=330 GPa
    #T_melt = T0*((P*1e-9-P0)/a+1.)**(1./c)
    ### Recalibrate to T_icb
    P_icb = 330e9
    T_icb = 5500.  #Hirose 2013: T_icb=5200-5700.   Hirose 2021 similar.
    T_melt = T_icb*( ((P*1e-9-P0)/a+1.)/((P_icb*1e-9-P0)/a+1.) )**(1./c)
    return(T_melt)

### EOS Vinet
def EOS_Vinet(rho,par):
    K0 = par[0]        #by setting these variables as parameters we are telling the function that they
    K0prime = par[1]   # can change and we can use set_f_params to tell the function which set
    gamma0 = par[2]    #of initial parameters it should use.
    gamma1 = par[3]
    alpha0 = par[4]
    alpha1 = par[5]
    rho0 = par[6]
    theta = ((3./2)*(K0prime-1))
    x=rho0/rho
    Kt = K0*x**(-2/3)*(1+(1+theta*x**(1/3))*(1-x**(1/3)))*np.exp(theta*(1-x**(1/3)))
    return(Kt)

### Density jump to new layer
def get_density_jump(rho,P,materials_new):
    K0_new = materials_new[0]
    K0prime_new = materials_new[1]
    K_T_next = K0_new + K0prime_new * P  #Poirier eq 4.8_ det _T for the first step in the new layer.
    #rho_arr = np.linspace(rho,rho*2.5,2000)  #arbitrary density range to find K_T_next, may have to be adjusted.
    #rho_arr = np.linspace(rho*0.8,rho*2.5,2000)  #arbitrary density range to find K_T_next, may have to be adjusted.
    rho_arr = np.linspace(rho*0.8,rho*2.0,4000)  #arbitrary density range to find K_T_next, may have to be adjusted.
    K_T_arr = EOS_Vinet(rho_arr,materials_new) #det K_T for all the possible rho of the array.
    i_rho_next = np.where(K_T_arr >= K_T_next)[0][0] #det which rho makes the new K_T equal to the one we expect for the layer. 
    rho_new = rho_arr[i_rho_next]
    delta_rho = rho_new-rho
    return(delta_rho)

### ODE derivatives for fundamental parameters.
def f_derivs(t,x,par):
    rho = x[0]         #dependent variables that are function of t
    m = x[1]           #t is required by the formalism of the function, in this case it stands 
    g = x[2]           #for the radius.
    P = x[3]
    T = x[4]
    
    gamma,alpha,Kt,Ks = get_agKtKs(rho,T,par)
    
    par[7] = alpha
    par[8] = gamma
    par[9] = Kt
    par[10] = Ks
    g_ref = par[11]
    r_ref = par[12]
    g_option = par[13]
    #g_values = [g,t*g_ref/r_ref]
    #g = g_values[g_option]
    #x[2] = g  #delete??
    
    drho = -(rho**2)*g/Ks
    dm = 4*np.pi*(t**2)*rho
    dg = dg_dr(t,m,rho,g_ref,r_ref,g_option)
    dP = -rho*g
    dT = -rho*g*gamma*T/Ks
    return ([drho, dm, dg, dP, dT])  #the order of the parameter in the return function should be the same
                                     #we used when setting the variables when defining the function.

### Get alpha,gamma,Kt,Ks
def get_agKtKs(rho,T,par):
    K0 = par[0]        #by setting these variables as parameters we are telling the function that they
    K0prime = par[1]   # can change and we can use set_f_params to tell the function which set
    gamma0 = par[2]    #of initial parameters it should use.
    gamma1 = par[3]
    alpha0 = par[4]
    alpha1 = par[5]
    rho0 = par[6]
    ratio_rho = rho0/rho  #corresponds to x in Boujibar et al. 2020
    gamma = gamma0*(ratio_rho**gamma1)
    alpha = alpha0*(ratio_rho**3)
    Kt = EOS_Vinet(rho,par)
    Ks = Kt*(1+(alpha*gamma*T))
    return(gamma,alpha,Kt,Ks)
    
### Gravity Function
def dg_dr(t,m,rho,g_ref,r_ref,g_option=0):
    dg_dr_arr=[4*np.pi*G*rho-(2*G*m)/(t**3),
            g_ref/r_ref]
    return dg_dr_arr[g_option]
    
### Get T_cmb
def get_T_cmb(f_heat,R_surface,m_planet,m_core,cp_mantle,cp_core,alpha_mantle,g_mantle,
                  args,liq_sol_mantle_ad_ratio=2.3,T_initial=255.):
    if args.earth:
        T_cmb = args.T_cmb
        T_um = args.T_um
    if not args.earth:
        cp_bulk = (m_core/m_planet)*cp_core + (1-m_core/m_planet)*cp_mantle
        DeltaT_accretion = (G*m_planet)/(cp_bulk*R_surface)
        DeltaT_G = f_heat*DeltaT_accretion
        R_cmb_guess = 3481e3*(m_planet/m_planet_E)**(0.245)## Valencia (06) fig 5c
        compressional_heat_fraction = liq_sol_mantle_ad_ratio*alpha_mantle*g_mantle/cp_mantle*(R_surface-R_cmb_guess)   #=DeltaT_ad/T_cmb fraction of T_cmb associated with adiabatic compression
        #print('liq_sol_mantle_ad_ratio=%.3f cp_bulk=%f DeltaT_G=%f R_cmb_guess=%f compressional_heat_fraction=%f alpha_mantle=%e DeltaT_acc=%f f_heat=%f'%(liq_sol_mantle_ad_ratio,cp_bulk,DeltaT_G,R_cmb_guess,compressional_heat_fraction,alpha_mantle,DeltaT_accretion,f_heat))
        #T_cmb = (T_initial+DeltaT_G)/(1-compressional_heat_fraction)
        T_cmb = (T_initial+DeltaT_G)/compressional_heat_fraction
        dTdr_ad = -alpha_mantle*g_mantle/cp_mantle*T_cmb
        r_um = R_surface  # T_um is mantle potential temp at surface
        T_um = T_cmb + dTdr_ad*(r_um-R_cmb_guess)
        ### Debugging:
        if T_cmb<0 or T_um<0:
            print('Error: T_um=%.1f T_cmb=%.1f cp_bulk=%.1f DeltaT_accretion=%.1f DeltaT_G=%.1f compr.frac.=%f R_surface=%.1f R_cmb_guess=%.1f'%(T_um,T_cmb,cp_bulk,DeltaT_accretion,DeltaT_G,compressional_heat_fraction,R_surface,R_cmb_guess))
    return(T_cmb,T_um)

### Get f_heat for early Earth.
def f_heat_Earth(liq_sol_mantle_ad_ratio=2.3):
    T_54 = 3300.  #fischer2015 @54 GPa
    T_initial = 255.  #initial T of accreting bodies
    g_54 = 10.04  #from E model at P=54 GPa
    rho_54 = 4749. #from E model at P=54 GPa
    r_54 = 5055404.
    r_cmb = 3481e3
    gamma_54 = 0.75
    alpha_54 = 1.08e-5
    Ks_54 = 373.49e9
    dT_ad_dr_solid = -rho_54*g_54*gamma_54*T_54/Ks_54
    dT_ad_dr_liq = liq_sol_mantle_ad_ratio*dT_ad_dr_solid  #factor of 2.3 is consistent with Andrault2011 fig 5, figuet2018 fig 4.3, stixrude 2009 fig 5.
    T_cmb_E_formation = T_54 + dT_ad_dr_liq*(r_cmb-r_54)
    from materials_earth import cp_mantle_bulk_E,cp_core_bulk_E
    cp_54 = cp_mantle_bulk_E
    m_core_E = m_planet_E-m_mantle_E
    cp_bulk = (m_core_E/m_planet_E)*cp_core_bulk_E + (1-m_core_E/m_planet_E)*cp_mantle_bulk_E
    compressional_heat_fraction = liq_sol_mantle_ad_ratio*alpha_54*g_54/cp_54*(R_surface_E-R_cmb)
    DeltaT_accretion = (G*m_planet_E)/(cp_bulk*R_surface_E)
    #f_heat_E = (1./DeltaT_accretion)*T_cmb_E_formation*(1-liq_sol_mantle_ad_ratio*alpha_54*g_54/cp_54*(R_surface-R_cmb))
    f_heat_E = (1./DeltaT_accretion)*\
        (T_cmb_E_formation*(compressional_heat_fraction)-T_initial)
    #print('T_cmb_E_formation=%.2f cp_bulk=%f DeltaT_acc=%f compressional_heat_fraction=%f'%(T_cmb_E_formation,cp_bulk,DeltaT_accretion,compressional_heat_fraction))
    return(f_heat_E)
    
    
    
############################################
### Input ###
def read_input_file(filename):
    f=open(filename,'r')
    input_file = f.read()
    f.close()
    lines = input_file.split('\n')
    input_parameters = [line for line in lines if not line.startswith('#')]
    ### Note (2/8/22): Never finished this bc input.py can just be imported.
    return()


############################################
### Output ###
### Write header to file
def write_output_header(f):
    f.write('i,r,rho,m,g,P,T,alpha,gamma,T_melt_Fe,Kt,Ks \n')
    return()

### Write data to file.
def write_output(f,state):
    ### state = [i,r,rho,m,g,P,T,alpha,gamma,T_melt_Fe,Kt,Ks]
    ### Note this should match header!!
    write_format = '%d,%f,%f,%e,%f,%e,%f,%e,%f,%f,%e,%e \n'
    f.write(write_format%tuple(state))
    return()

### Read output file
def read_data(filename):
    #nlayers=5
    print('Reading '+filename+' ...')
    f=open(filename,'r')
    ### Note: each out file contains a header line with parameter names, a line with ir_layers, and
    ### then a big list of the parameters
    runname = f.readline().split('=')[1].split(' ')[0]
    layers_found = f.readline().split('=')[1].split(' ')[0][1:-1].split(',')
    phases_found = f.readline().split('=')[1].split(' ')[0][1:-1].split(',')
    ir_comp_layer = eval(f.readline().split('=')[1])
    ir_phase_layer = eval(f.readline().split('=')[1])
    glob_variab = eval(f.readline().split(' \n')[0].split('=')[1])
    i_g_change = int(f.readline().split('=')[1].split(' ')[0])
    labels = f.readline().split(' ')[0].split(',')
    n = 0
    line = ''
    data = dict.fromkeys(labels,np.zeros(1))
    reading = True
    line = f.readline()
    while line:
        nums = line.split(' ')[:-1][0].split(',')
        for i in range(len(labels)):
            data[labels[i]] = np.append(data[labels[i]],float(nums[i]))
        n+=1
        line = f.readline()
    ### Remove intial zero from data dict arrays
    for label in labels: data[label]=data[label][1:]
    return(runname,layers_found,phases_found,ir_comp_layer,ir_phase_layer,glob_variab,i_g_change,labels,data)



### Write data to file.
def write_run_to_file(runname,layers_found,phases_found,ir_comp_layer,ir_phase_layer,glob_variab,i_g_change,r,rho,m,g,P,T,alpha,gamma,T_melt_Fe,Kt,Ks):
    outputfile = runname+'.out'
    f=open(outputfile,'w')
    f.write('runname='+runname+' \n')
    f.write('layers_found=['+','.join(layers_found)+'] \n')
    f.write('phases_found=['+','.join(phases_found)+'] \n')   
    f.write('ir_comp_layer=['+','.join(str(x) for x in ir_comp_layer)+'] \n')
    f.write('ir_phase_layer=['+','.join(str(x) for x in ir_phase_layer)+'] \n')    
    f.write('glob_variab=['+','.join([str(y) for y in glob_variab])+'] \n')
    f.write('i_g_change=%i \n'%i_g_change)
    write_output_header(f)    
    for j in range(len(r)):   ### loop through each layer and write to outputfile
        state=[j,r[j],rho[j],m[j],g[j],P[j],T[j],alpha[j],gamma[j],T_melt_Fe[j],Kt[j],Ks[j]]
        write_output(f,state)
    f.close()  #close file
    print('wrote data to '+outputfile)
    return()

#### clean_pycache()
def clean_pycache():
    import pathlib
    [p.unlink() for p in pathlib.Path('./__pycache__').rglob('*')]
    [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]
    return()



########  PLOTTING #######

def plot_profiles(r,m,g,P,rho,m_planet):
    ### Find radial index of 660km
    r_660_E = 6371e3-660e3  #radius of 660 km depth in E.
    i_660 = np.argmin(np.abs(r-r_660_E))
    ### Begin Plots ###
    ### 4x1 Profiles
    plt.figure(figsize= (6,10))
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
    plt.ylim(0,np.min([20,max(g)]))
    ### P(r)
    plt.subplot(413)
    plt.plot(r*1e-3,P*1e-9)
    plt.xlabel('r');plt.ylabel('Pressure (GPa)')
    plt.ylim(0,np.min([400,max(P)*1e-9]))
    ### rho(r)
    plt.subplot(414)
    plt.plot(r*1e-3,rho)
    plt.plot(r[i_660]*1e-3,rho_660_pd,'o')
    plt.plot(r[i_660]*1e-3,rho_660_pv,'o')
    plt.xlabel('r');plt.ylabel('density (kg/m^3)')
    plt.ylim(0,np.min([15e3,max(rho)]))
    plt.tight_layout()

def plot_thermo_profiles(r,rho,T,T_melt_Fe,gamma,i_cmb):
    ### 3x1 Thermodynamic Profiles
    plt.figure(figsize= (6,10))
    plt.subplots_adjust(hspace=.5)
    ### rho(r)
    plt.subplot(311)
    plt.plot(r*1e-3,rho)
    plt.xlabel('r');plt.ylabel('rho')
    ### T(r)
    plt.subplot(312)
    plt.plot(r*1e-3,T)
    #plt.plot(r*1e-3,T_melt_Fe,color='red')
    plt.plot(r[i_cmb:]*1e-3,T_melt_Fe[i_cmb:],color='red')
    #plt.xlim(0,3481)
    plt.xlabel('r');plt.ylabel('T')
    ### Gamma(r)
    plt.subplot(313)
    plt.plot(r*1e-3,gamma)
    plt.xlabel('r');plt.ylabel('gamma')
    plt.tight_layout()
    return()

### Compare to prem
def compare_to_prem(r,m,rho,P,T,gamma,ir_phase_layer,i_cmb,i_icb):
    R_surface = r[0]
    #R_cmb = r[i_cmb]
    #R_icb = r[i_icb]
    j=1
    print("pd to pv  : r=%.0f km z=%.0f km P=%.2f GPa rho(above,below)=(%.0f,%.0f) change rho=%f "%(r[ir_phase_layer[j]]*1e-3,(R_surface-r[ir_phase_layer[j]])*1e-3,P[ir_phase_layer[j]]*1e-9,rho[ir_phase_layer[j]],rho[ir_phase_layer[j]+1],(rho[ir_phase_layer[j]+1]/rho[ir_phase_layer[j]]-1)))
    print("PREM      : r=%.0f km z=%.0f km P=%.2f GPa rho(above,below)=(%.0f,%d) change rho=%f "\
              %(R_660*1e-3,(R_surface-R_660)*1e-3,P_660*1e-9,rho_660_pd,\
              rho_660_pv,(rho_660_pv/rho_660_pd-1)))
    j=j+1
    print("pv to ppv : r=%d km z=%d km P=%.2f GPa rho(above,below)=(%d,%d) change rho=%f "%(r[ir_phase_layer[j]]*1e-3,(R_surface-r[ir_phase_layer[j]])*1e-3,P[ir_phase_layer[j]]*1e-9,rho[ir_phase_layer[j]],rho[ir_phase_layer[j]+1],(rho[ir_phase_layer[j]+1]/rho[ir_phase_layer[j]]-1)))
    j=j+1
    print("CMB : r=%d km z=%d km P=%.2f GPa rho(above,below)=(%d,%d) change rho=%f "%\
              (r[i_cmb]*1e-3,(R_surface-r[i_cmb])*1e-3,P[i_cmb]*1e-9,\
                   rho[i_cmb-1],rho[i_cmb+1],(rho[i_cmb+1]/rho[i_cmb-1]-1)))
    print("PREM: r=%d km z=%d km P=%.2f GPa rho(above,below)=(%d,%d) change rho=%f "\
              %(R_cmb*1e-3,(R_surface-R_cmb)*1e-3,P_cmb*1e-9,rho_cmb_man,\
              rho_cmb_core,(rho_cmb_core/rho_cmb_man-1)))
    j=j+1
    print("ICB : r=%d km z=%d km P=%.2f GPa rho(above,below)=(%d,%d) change rho=%f "%\
              (r[i_icb]*1e-3,(R_surface-r[i_icb])*1e-3,P[i_icb]*1e-9,\
                   rho[i_icb],rho[i_icb+1],(rho[i_icb+1]/rho[i_icb]-1)))
    print("PREM: r=%d km z=%d km P=%.2f GPa rho(above,below)=(%d,%d) change rho=%f "\
              %(R_icb*1e-3,(R_surface-R_icb)*1e-3,P_icb*1e-9,rho_icb_oc,\
              rho_icb_ic,(rho_icb_ic/rho_icb_oc-1)))
    i_1221 = np.argmin(np.abs(r-R_icb))
    print("1221 km: rho=%d T=%.2f gamma=%.4f"%(rho[i_1221],T[i_1221],gamma[i_1221]))
    print("center : r=%.2f km P=%.2f GPa rho=%d m=%e "%\
              (r[-1]*1e-3,P[-1]*1e-9,rho[-1],m[-1]))
