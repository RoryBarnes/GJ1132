import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import vplot

sKey = "b,Age,final"
sPlotFile = "EnvLossQuiescentHist.png"
iNumBins = 50
dFigSizeX = 6.5
dFigSizeY = 7


dYearSec = 3600*24*365.25
dLowerBoundAge = 0
dUpperBoundAge = 1e9*dYearSec
dGyr=1e9*dYearSec

def LoadData(file):
    with open(file, 'r') as file:
        content = file.read().strip()  # Read and remove surrounding whitespace
        
        # Check if content is wrapped in double quotes
        try:
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]  # Remove the leading and trailing quotes
                content = content.replace('\\"', '"')  # Replace escaped quotes with normal quotes
        except:
            print("ERROR: Unable to load file "+file)
            exit()

        try:
            data = json.loads(content)  # Now load as JSON
            #print("Loaded JSON data:", data)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            exit()
    return data

def ExtractColumn(data,sKey,dLowerBound,dUpperBound):
    if isinstance(data, dict):  # Check if the loaded content is a dictionary
        daColumn = data.get(sKey)
        daColumnClean = pd.Series(daColumn).dropna()
        daColumnFiltered = np.array(daColumnClean)
        daColumnBound = daColumnFiltered[(daColumnFiltered >= dLowerBound) & (daColumnFiltered <= dUpperBound)]

        return daColumnBound
    else:
        print("ERROR: Loaded data is not a dictionary.")
        exit()

def NormalizedHistogram(column,color,label=None):
    column_clean = pd.Series(column).dropna()
    counts, bin_edges = np.histogram(column_clean, bins=iNumBins)

    # Normalize the counts to fractions
    fractions = counts / len(column_clean)

    # Clear the figure and re-plot the normalized histogram
    #ax[x,y].clear()  # Clear the previous plot
    plt.step(bin_edges[:-1], fractions,where='mid', color=color,label=label)

daDataQuiescent = LoadData("Quiescent/output/Converged_Param_Dictionary.json")
daAgeQuiescent = ExtractColumn(daDataQuiescent,sKey,dLowerBoundAge,dUpperBoundAge)/dGyr

daDataFlares = LoadData("Flares/output/Converged_Param_Dictionary.json")
daAgeFlares = ExtractColumn(daDataFlares,sKey,dLowerBoundAge,dUpperBoundAge)/dGyr

daDataCME = LoadData("CME/output/Converged_Param_Dictionary.json")
daAgeCME = ExtractColumn(daDataCME,sKey,dLowerBoundAge,dUpperBoundAge)/dGyr


#if daEnvelopeMass.size > 0 and daAge.size > 0:
if daAgeQuiescent.size > 0 and daAgeFlares.size > 0 and daAgeCME.size > 0:
    fig = plt.figure(figsize=(dFigSizeX, dFigSizeY))
    
    NormalizedHistogram(daAgeQuiescent,vplot.colors.pale_blue,label='Quiescent')
    NormalizedHistogram(daAgeFlares,vplot.colors.orange,label='w/Flares')
    NormalizedHistogram(daAgeCME,'k',label='w/Flares+CMEs')
    plt.xlabel('Time to Remove Envelope [Gyr]',fontsize=18)
    plt.ylabel('Fraction',fontsize=18)
    plt.xlim(0,1)
    plt.ylim(0,0.1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best')
    plt.tight_layout()

    #plt.show()
    plt.savefig(sPlotFile,dpi=300)
else:
    print("ERROR: At least one array does not exist.")
    exit()
