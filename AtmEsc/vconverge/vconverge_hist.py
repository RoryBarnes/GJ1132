import json
import pandas as pd
import matplotlib.pyplot as plt

file=open("500TO/Converged_Param_Dictionary.json")

data = json.load(file)

daWaterFinal=data["b,SurfWaterMass,final"]

#print(daWaterFinal)
#print(daWaterFinal[0])

plt.hist(daWaterFinal)
plt.xlabel('Final Water (TO)')
plt.ylabel('Number')
plt.show()

file.close()
