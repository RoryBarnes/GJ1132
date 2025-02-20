import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import vplot

co2solfactor = 1e18
co2mofactor = 1e21

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
    counts, bin_edges = np.histogram(column_clean, bins=iNumBins)

    # Normalize the counts to fractions
    fractions = counts / len(column_clean)

    # Clear the figure and re-plot the normalized histogram
    #ax[x,y].clear()  # Clear the previous plot
    ax[x,y].step(bin_edges[:-1], fractions,where='mid', color=color,label=label)

both = GetData("RadTidal/output/Converged_Param_Dictionary.json")
daSurfTempBoth = ExtractColumn(both,"b,SurfTemp,final")
daWaterSolBoth = ExtractColumn(both,"b,WaterMassSol,final")
daWaterMOBoth = ExtractColumn(both,"b,WaterMassMOAtm,final")
daWaterAtmBoth = ExtractColumn(both,"b,PressWaterAtm,final")
daOxyAtmBoth = ExtractColumn(both,"b,PressOxygenAtm,final")
daFe2O3Both = ExtractColumn(both,"b,FracFe2O3Man,final")
daCO2SolBoth = ExtractColumn(both,"b,CO2MassSol,final")
daCO2MOBoth = ExtractColumn(both,"b,CO2MassMOAtm,final")
daCO2AtmBoth = ExtractColumn(both,"b,PressCO2Atm,final")

rad = GetData("RadOnly/output/Converged_Param_Dictionary.json")
daSurfTempRad = ExtractColumn(rad,"b,SurfTemp,final")
daWaterSolRad = ExtractColumn(rad,"b,WaterMassSol,final")
daWaterMORad = ExtractColumn(rad,"b,WaterMassMOAtm,final")
daWaterAtmRad = ExtractColumn(rad,"b,PressWaterAtm,final")
daOxyAtmRad = ExtractColumn(rad,"b,PressOxygenAtm,final")
daFe2O3Rad = ExtractColumn(rad,"b,FracFe2O3Man,final")
daCO2SolRad = ExtractColumn(rad,"b,CO2MassSol,final")
daCO2MORad = ExtractColumn(rad,"b,CO2MassMOAtm,final")
daCO2AtmRad = ExtractColumn(rad,"b,PressCO2Atm,final")

iNumBins = 50

fig, ax = plt.subplots(nrows = 3, ncols=3, figsize=(6.5, 7))
NormHistPanel(daSurfTempBoth,'k',0,0)
NormHistPanel(daSurfTempRad,vplot.colors.pale_blue,0,0)
ax[0,0].set_xlabel('Surface Temp [K]')
ax[0,0].set_ylabel('Number')

NormHistPanel(np.log10(daFe2O3Both),'k',0,1,label='Rad+Tides')
NormHistPanel(np.log10(daFe2O3Rad),vplot.colors.pale_blue,0,1,label='Rad Only')
ax[0,1].legend()
ax[0,1].set_xlabel(r'log$_{10}$(Fe$_2$O$_3$ Fraction)')
ax[0,1].set_ylabel('Fraction')    

NormHistPanel(daOxyAtmBoth,'k',0,2)
NormHistPanel(daOxyAtmRad,vplot.colors.pale_blue,0,2)
ax[0,2].set_xlabel('Oxygen Atm. [bar]')
ax[0,2].set_ylabel('Fraction')

NormHistPanel(daWaterMOBoth,'k',1,0)
NormHistPanel(daWaterMORad,vplot.colors.pale_blue,1,0)
ax[1,0].set_xlabel('Water Magma Ocean [TO]')
ax[1,0].set_ylabel('Fraction')

NormHistPanel(daWaterSolBoth,'k',1,1)
NormHistPanel(daWaterSolRad,vplot.colors.pale_blue,1,1)
ax[1,1].set_xlabel('Water Mantle [TO]')
ax[1,1].set_ylabel('Fraction')

NormHistPanel(daWaterAtmBoth,'k',1,2)
NormHistPanel(daWaterAtmRad,vplot.colors.pale_blue,1,2)
ax[1,2].set_xlabel('Water Atm [bar]')
ax[1,2].set_ylabel('Fraction')

NormHistPanel(daCO2MOBoth/co2mofactor,'k',2,0)
NormHistPanel(daCO2MORad/co2mofactor,vplot.colors.pale_blue,2,0)
ax[2,0].set_xlabel(r'CO$_2$ Magma Ocean [$10^{21}$kg]')
ax[2,0].set_ylabel('Fraction')

NormHistPanel(daCO2SolBoth/co2solfactor,'k',2,1)
NormHistPanel(daCO2SolRad/co2solfactor,vplot.colors.pale_blue,2,1)
ax[2,1].set_xlabel(r'CO$_2$ Mantle [$10^{18}$kg]')
ax[2,1].set_ylabel('Fraction')

NormHistPanel(daCO2AtmBoth,'k',2,2)
NormHistPanel(daCO2AtmRad,vplot.colors.pale_blue,2,2)
ax[2,2].set_xlabel(r'CO$_2$ Atm. [bar]')
ax[2,2].set_ylabel('Fraction')


plt.savefig('GJ1132_MagmaOcean_Dist.png',dpi=300)

