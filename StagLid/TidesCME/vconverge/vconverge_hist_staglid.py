import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import vplot

co2solfactor = 1e18
co2mofactor = 1e21
gyearsec = 3600*25*365.25*1e9

def GetData(filename):
    with open(filename, 'r') as file:
        content = file.read().strip()  # Read and remove surrounding whitespace
        
        # Check if content is wrapped in double quotes
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]  # Remove the leading and trailing quotes
            content = content.replace('\\"', '"')  # Replace escaped quotes with normal quotes

        try:
            data = json.loads(content)  # Now load as JSON
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
        return data

def ExtractColumn(data,key):
    if isinstance(data, dict):  # Check if the loaded content is a dictionary
        return np.array(data.get(key))

def NormHistPanel(column,color,x,y,label=None):
    column_clean = pd.Series(column).dropna()
    column_filtered = column_clean[(column_clean >= 0)]
    counts, bin_edges = np.histogram(column_filtered, bins=iNumBins)

    # Normalize the counts to fractions
    fractions = counts / len(column_filtered)
    #print(column_filtered)
    # Clear the figure and re-plot the normalized histogram
    #ax[x,y].clear()  # Clear the previous plot
    ax[x,y].step(bin_edges[:-1], fractions, where='mid', color=color,label=label)

#out = GetData("vconverge_tmp/tmp_Converged_Param_Dictionary.json")
out = GetData("Unconverged_Param_Dictionary.json")
daWaterAtm = ExtractColumn(out,"b,SurfWaterMass,final")
daWaterCrust = ExtractColumn(out,"b,CrustWaterMass,final")
daWaterMan = ExtractColumn(out,"b,ManWaterMass,final")
daCO2Atm = ExtractColumn(out,"b,SurfCO2Mass,final")
daCO2Man = ExtractColumn(out,"b,ManCO2Mass,final")
daCO2Crust = ExtractColumn(out,"b,CrustCO2Mass,final")
daOxyAtm = ExtractColumn(out,"b,OxygenMass,final")
daTMan = ExtractColumn(out,"b,MantleTemp,final")
daMagMom = ExtractColumn(out,"b,MagMom,final")

iNumBins = 50

fig, ax = plt.subplots(nrows = 3, ncols=3, figsize=(6.5, 7))
# NormHistPanel(daSurfTemp,'k',0,0)
# ax[0,0].set_xlabel('Surface Temp [K]')
# ax[0,0].set_ylabel('Number')
NormHistPanel(daWaterAtm,'k',0,0)
ax[0,0].set_xlabel('Water Atm [bar]')
ax[0,0].set_ylabel('Fraction')
#ax[0,0].set_title('Magma Ocean')

#NormHistPanel(np.log10(daFe2O3),'k',0,1)
#ax[0,1].set_xlabel(r'log$_{10}$(Fe$_2$O$_3$ Fraction)')
NormHistPanel(daWaterCrust,'k',0,1)
ax[0,1].set_xlabel(r'Water Crust [TO]')
ax[0,1].set_ylabel('Fraction')    
#ax[0,1].set_title('Mantle')

NormHistPanel(daWaterMan,'k',0,2)
ax[0,2].set_xlabel('Water Mantle [TO]')
ax[0,2].set_ylabel('Fraction')
#ax[0,2].set_title('Atmosphere')

daCO2Atm = daCO2Atm[(daCO2Atm >= 0) & (daCO2Atm <= 100)]

NormHistPanel(daCO2Atm,'k',1,0)
ax[1,0].set_xlabel(r'CO$_2$ Atm [bar]')
ax[1,0].set_ylabel('Fraction')
#print(daCO2Atm)

NormHistPanel(daCO2Crust,'k',1,1)
ax[1,1].set_xlabel(r'CO$_2$ Crust [bar]')
ax[1,1].set_ylabel('Fraction')

NormHistPanel(daCO2Man,'k',1,2)
ax[1,2].set_xlabel(r'CO$_2$ Mantle [bar]')
ax[1,2].set_ylabel('Fraction')

NormHistPanel(daOxyAtm,'k',2,0)
ax[2,0].set_xlabel('Oxygen Atm. [bar]')
ax[2,0].set_ylabel('Fraction')

NormHistPanel(daTMan,'k',2,1)
ax[2,1].set_xlabel('Mantle Temp. [K]')
ax[2,1].set_ylabel('Fraction')

NormHistPanel(daCO2Atm/1e3,'k',2,2)
ax[2,2].set_xlabel(r'Mag. Moment [$\oplus$ Units]')
ax[2,2].set_ylabel('Fraction')


plt.savefig('GJ1132_StagLid_Dist.png',dpi=300)

