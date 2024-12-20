import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import vplot

dYearSec = 3600*24*365.25
# Step 1: Load and process JSON file
def LoadData():
    with open("output/Converged_Param_Dictionary.json", 'r') as file:
        content = file.read().strip()  # Read and remove surrounding whitespace
        
        # Check if content is wrapped in double quotes
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]  # Remove the leading and trailing quotes
            content = content.replace('\\"', '"')  # Replace escaped quotes with normal quotes

        try:
            data = json.loads(content)  # Now load as JSON
            #print("Loaded JSON data:", data)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            exit()
    return data

# Step 2: Extract the array using the key
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

def NormHist(column,color,label=None):
    column_clean = pd.Series(column).dropna()
    counts, bin_edges = np.histogram(column_clean, bins=iNumBins)

    # Normalize the counts to fractions
    fractions = counts / len(column_clean)

    # Clear the figure and re-plot the normalized histogram
    #ax[x,y].clear()  # Clear the previous plot
    plt.step(bin_edges[:-1], fractions,where='mid', color=color,label=label)

data = LoadData()

# sKey = "b,EnvelopeMass,final"
# dLowerBoundMass = 0
# dUpperBoundMass = 1
# daEnvelopeMass = ExtractColumn(data,sKey,dLowerBoundMass,dUpperBoundMass)

sKey = "b,Age,final"
dLowerBoundAge = 0
dUpperBoundAge = 1e9*dYearSec
dGyr=1e9*dYearSec
daAge = ExtractColumn(data,sKey,dLowerBoundAge,dUpperBoundAge)/dGyr


#if daEnvelopeMass.size > 0 and daAge.size > 0:
if daAge.size > 0:
    iNumBins = 50  # Adjust as needed
    fig = plt.figure(figsize=(6.5, 7))
    
    NormHist(daAge,'k')
    #ax[1].hist(daAge,bins=iNumBins)
    plt.xlabel('Time to Remove Envelope [Gyr]',fontsize=18)
    plt.ylabel('Fraction',fontsize=18)
    #ax[1].set_xlim(dLowerBoundAge/dGyr,dUpperBoundAge/dGyr)
    plt.xlim(0,1)
    plt.ylim(0,0.1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    #plt.show()
    plt.savefig('EnvLossQuiescentHist.png',dpi=300)
else:
    print("ERROR: At least one array does not exist.")
    exit()
