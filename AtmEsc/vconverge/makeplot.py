import matplotlib.pyplot as plt 
import vplot as vpl
import json
import numpy as np

saDirs = ["300TO", "400TO", "500TO", 
            "600TO", "700TO", "800TO", "900TO", "1000TO", 
            "1100TO", "1200TO", "1300TO", "1400TO", "1500TO", 
            "1600TO", "1700TO", "1800TO", "1900TO", "2000TO", 
            "2100TO", "2200TO", "2300TO", "2400TO", "2500TO", 
            "2600TO", "2700TO", "2800TO", "2900TO", "3000TO",
            "3100TO", "3200TO", "3300TO", "3500TO"]

daProb = [0 for i in range(len(saDirs))]

fig = plt.figure(figsize=(6.5, 4))
plt.subplot(1, 2, 1)
iTrial = 0
for sDir in saDirs:
    print(sDir)
    sFile=sDir+"/Converged_Param_Dictionary.json"
    iLength=len(sDir)-2
    dWaterInitial = int(sDir[0:iLength])
    sJson=open(sFile)

    data = json.load(sJson)

    daWaterFinal=data["b,SurfWaterMass,final"]
    iNumTrials = len(daWaterFinal)
    daWaterInitial = [dWaterInitial for i in range(iNumTrials)]
    iNumDevolatilized = 0
    for dWaterFinal in daWaterFinal:
        #print(dWaterFinal)
        if dWaterFinal == 0.0:
            iNumDevolatilized += 1
    #print(iNumDevolatilized)
    daProb[iTrial] = iNumDevolatilized/iNumTrials
    iTrial += 1

    plt.scatter(daWaterInitial,np.array(daWaterFinal)/np.array(daWaterInitial),color='black', alpha=0.1, s=10)

    plt.xlabel("Iniital Water (TO)")
    plt.ylabel("Fraction Water Remaining")

iTrial = 0
daWaterInitial = [0 for i in range(len(saDirs))]

for sDir in saDirs:
    iLength=len(sDir)-2
    daWaterInitial[iTrial] = int(sDir[0:iLength])
    iTrial += 1

plt.subplot(1, 2, 2)

plt.plot(daWaterInitial,daProb,color='black')
plt.ylim(0,1)
plt.xlabel("Initial Water (TO)")
plt.ylabel("P(Devolatilized)")

fig.savefig('gj1132b_devolatization.pdf')
