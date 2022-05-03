import matplotlib.pyplot as plt 
import vplot as vpl
import json

saDirs = ["500TO", "1000TO"]

daProb = [0 for i in range(len(saDirs))]

fig = plt.figure(figsize=(8.5, 6))
plt.subplot(2, 1, 1)
iTrial = 0
for sDir in saDirs:
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
        print(dWaterFinal)
        if dWaterFinal == 0.0:
            iNumDevolatilized += 1
    print(iNumDevolatilized)
    daProb[iTrial] = iNumDevolatilized/iNumTrials
    iTrial += 1

    plt.plot(daWaterInitial,daWaterFinal,color='black')

    plt.xlabel("Iniital Water (TO)")
    plt.ylabel("Final Water (TO)")

iTrial = 0
daWaterInitial = [0 for i in range(len(saDirs))]

for sDir in saDirs:
    iLength=len(sDir)-2
    daWaterInitial[iTrial] = int(sDir[0:iLength])
    iTrial += 1

plt.subplot(2, 1, 2)

plt.plot(daWaterInitial,daProb)
plt.xlabel("Initial Water (TO)")
plt.ylabel("P(Devolatilized")

fig.savefig('gj1132b_devolatization.pdf')
