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
daAge = ExtractColumn(out,"Age,final")/gyearsec
daSurfTemp = ExtractColumn(out,"b,SurfTemp,final")
daWaterSol = ExtractColumn(out,"b,WaterMassSol,final")
daWaterMO = ExtractColumn(out,"b,WaterMassMOAtm,final")
daWaterAtm = ExtractColumn(out,"b,PressWaterAtm,final")
daOxyAtm = ExtractColumn(out,"b,PressOxygenAtm,final")
daFe2O3 = ExtractColumn(out,"b,FracFe2O3Man,final")
daCO2Sol = ExtractColumn(out,"b,CO2MassSol,final")
daCO2MO = ExtractColumn(out,"b,CO2MassMOAtm,final")
daCO2Atm = ExtractColumn(out,"b,PressCO2Atm,final")

iNumBins = 50

fig, ax = plt.subplots(nrows = 3, ncols=3, figsize=(6.5, 7))
# NormHistPanel(daSurfTemp,'k',0,0)
# ax[0,0].set_xlabel('Surface Temp [K]')
# ax[0,0].set_ylabel('Number')
NormHistPanel(daAge,'k',0,0)
ax[0,0].set_xlabel('Solidification Age [Gyr]')
ax[0,0].set_ylabel('Fraction')
ax[0,0].set_title('Magma Ocean')

#NormHistPanel(np.log10(daFe2O3),'k',0,1)
#ax[0,1].set_xlabel(r'log$_{10}$(Fe$_2$O$_3$ Fraction)')
NormHistPanel(daFe2O3,'k',0,1)
ax[0,1].set_xlabel(r'Mantle Fe$_2$O$_3$ Fraction')
ax[0,1].set_ylabel('Fraction')    
ax[0,1].set_title('Mantle')

NormHistPanel(daOxyAtm/1e3,'k',0,2)
ax[0,2].set_xlabel('Oxygen Atm. [kbar]')
ax[0,2].set_ylabel('Fraction')
ax[0,2].set_title('Atmosphere')

NormHistPanel(daWaterMO,'k',1,0)
ax[1,0].set_xlabel('Water Magma Ocean [TO]')
ax[1,0].set_ylabel('Fraction')

daWaterSol = daWaterSol[(daWaterSol >= 0) & (daWaterSol <= 5)]

NormHistPanel(daWaterSol,'k',1,1)
ax[1,1].set_xlabel('Water Mantle [TO]')
ax[1,1].set_ylabel('Fraction')

daWaterAtm = daWaterAtm[(daWaterAtm >= 0) & (daWaterAtm <= 5e4)]

NormHistPanel(daWaterAtm/1e3,'k',1,2)
ax[1,2].set_xlabel('Water Atm [kbar]')
ax[1,2].set_ylabel('Fraction')

NormHistPanel(daCO2MO/co2mofactor,'k',2,0)
ax[2,0].set_xlabel(r'CO$_2$ Magma Ocean [$10^{21}$kg]')
ax[2,0].set_ylabel('Fraction')

daCO2Sol = daCO2Sol[(daCO2Sol >= 0) & (daCO2Sol <= 1e21)]

NormHistPanel(daCO2Sol/co2solfactor,'k',2,1)
ax[2,1].set_xlabel(r'CO$_2$ Mantle [$10^{18}$kg]')
ax[2,1].set_ylabel('Fraction')

NormHistPanel(daCO2Atm/1e3,'k',2,2)
ax[2,2].set_xlabel(r'CO$_2$ Atm. [kbar]')
ax[2,2].set_ylabel('Fraction')


plt.savefig('GJ1132_MagmaOcean_Dist.png',dpi=300)

