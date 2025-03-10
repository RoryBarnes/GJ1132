import numpy as np
import matplotlib.pyplot as plt
import vplanet
import vplot as vpl

cpl = vplanet.get_output("CPL")
ctl = vplanet.get_output("CTL")
maxwell = vplanet.get_output("Maxwell")

fig = plt.figure(figsize=(6.5, 4))

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Eccenricity",fontsize=20)
plt.xlabel("Time [yr]",fontsize=20)
plt.plot(cpl.b.Time,cpl.b.Eccentricity,color=vpl.colors.red,label='CPL')
plt.plot(ctl.b.Time,ctl.b.Eccentricity,color=vpl.colors.purple,label='CTL')
plt.axhline(y=0, color=vpl.colors.purple,xmin=0.67,xmax=1)
plt.plot(maxwell.b.Time,maxwell.b.Eccentricity,color=vpl.colors.pale_blue,label='Maxwell')
plt.legend(loc='best',fontsize=16)

plt.xscale('log')

#plt.show()
plt.savefig('GJ1132bEccTideComp.png',dpi=300)