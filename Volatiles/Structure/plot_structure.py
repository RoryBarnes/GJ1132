import matplotlib.pyplot as plt
import numpy as np
import vplot
import csv

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

file = "input_Earth1.6_cmf0.05.out"
### Read in data
runname,layers_found,phases_found,ir_comp_layer,ir_phase_layer,glob_variab,i_g_change,labels,data = read_data(file)

r = data['r']
m = data['m']
rho = data['rho']
press = data['P']
temp = data['T']
m_planet = m[0]
R_surface = r[0]

file = "input_Earth.out"
### Read in data
erunname,elayers_found,ephases_found,eir_comp_layer,eir_phase_layer,eglob_variab,ei_g_change,elabels,edata = read_data(file)



er = edata['r']
em = edata['m']
erho = edata['rho']
epress = edata['P']
etemp = edata['T']
em_planet = em[0]
eR_surface = er[0]
mearth = 5.972186e24
rearth = 6.3781e6

file = "PREM_1s.csv"
premr = [0 for i in range(199)]
premd = [0 for i in range(199)]
i=0
with open(file, newline='') as csvfile:
    premreader = csv.reader(csvfile, delimiter=',')
    for row in premreader:
        premr[i]=float(row[0])*1000/rearth
        premd[i]=float(row[2])*1000
        i = i+1

print(em_planet,m_planet)

nfig=1
plt.figure(nfig,figsize= (6.5,4))
plt.subplots_adjust(hspace=.5)
### M(r)
plt.subplot(221)
plt.plot(r/rearth,m/mearth,color=vplot.colors.red)
plt.plot(er/rearth,em/mearth,color=vplot.colors.pale_blue)
plt.xlabel(r'Radius (R$_\oplus$)')
plt.ylabel(r'Mass (M$_\oplus$)')
plt.ylim(0,1.61)
plt.xlim(0,1.2)

plt.subplot(222)
plt.plot(r/rearth,rho,color=vplot.colors.red,label="GJ 1132 b")
plt.plot(er/rearth,erho,color=vplot.colors.pale_blue,label="Earth")
plt.plot(premr,premd,color=vplot.colors.dark_blue,linestyle='dotted',label="PREM")
#plt.plot(premr,premd,color='b')
plt.xlabel(r'Radius (R$_\oplus$)')
plt.ylabel(r'Density (kg/m$^3$)')
plt.ylim(2000,14000)
plt.xlim(0,1.2)
plt.legend(loc="best")

plt.subplot(223)
plt.plot(r/rearth,temp,color=vplot.colors.red)
plt.plot(er/rearth,etemp,color=vplot.colors.pale_blue)
plt.xlabel(r'Radius (R$_\oplus$)')
plt.ylabel('Temperature (K)')
plt.xlim(0,1.2)

### P(r)
plt.subplot(224)
plt.plot(r/rearth,press*1e-9,color=vplot.colors.red)
plt.plot(er/rearth,epress*1e-9,color=vplot.colors.pale_blue)
plt.xlabel(r'Radius (R$_\oplus$)')
plt.ylabel('Pressure (GPa)')
plt.ylim(0,400)
plt.xlim(0,1.2)





plt.tight_layout()
#plt.savefig('structure.png')
plt.savefig('structure.pdf')
