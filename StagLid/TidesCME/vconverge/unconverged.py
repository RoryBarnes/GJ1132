import json
import pandas as pd
import numpy as np
from pathlib import Path

body = 'b'
vplanet_logfile = 'gj1132.log'

YearSec = 365.25*3600*24
TO=1.39e21


StopTime = []
Time = []

WaterManConc = []
CrustWaterMass = []
SurfWaterMass = []

OxygenMass = []

SurfCO2Mass = []
CrustCO2Mass = []
ManCO2Mass = []

subdirs = [str(Path("output") / d.name) for d in Path("output").iterdir() if d.is_dir()]

missing = 0
found = 0
for subdir in subdirs:

    file_path = Path(subdir) / vplanet_logfile

    if not file_path.exists():
        print(f"File {file_path} not found! Ignoring.")
        missing += 1
        continue  # or continue if in a loop

    #print(file_path)
    try:
        with open(file_path, 'r') as curr_file:
            curr_lines = curr_file.readlines()
            
        final = 0
        ready = 0
        system = 0
        co2masssol = 0
        found += 1        
        for line in curr_lines:
            parts = line.split()
            if len(parts) <= 2:
                continue
            
            if parts[0] == 'Stop' and parts[1] == 'Time:':
                StopTime.append(float(parts[2]))

            if parts[1] == 'FINAL':
                final = 1
            
            if final and parts[2] == 'SYSTEM':
                system = 1
            
            if final and parts[1] == 'BODY:' and parts[2] == body:
                ready = 1

            if ready and parts[0] == '(SurfWaterMass)':
                SurfWaterMass.append(float(parts[-1]))
            
            if ready and parts[0] == '(ManCO2Mass)':
                ManCO2Mass.append(float(parts[-1]))

            if ready and parts[0] == '(CrustWaterMass)':
                CrustWaterMass.append(float(parts[-1]))

            if ready and parts[0] == '(CrustCO2Mass)':
                CrustCO2Mass.append(float(parts[-1]))

            if ready and parts[0] == '(OxygenMass)':
                OxygenMass.append(float(parts[-1]))

            if ready and parts[0] == '(SurfaceCO2Mass)':
                SurfCO2Mass.append(float(parts[-1]))

            if ready and parts[0] == '(WaterManConc)':
                WaterManConc.append(float(parts[-1]))

            if final and parts[0] == '(Time)':
                Time.append(float(parts[-1]))

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

output = {
    "b,SurfWaterMass,final": SurfWaterMass,
    "b,CrustWaterMass,final": CrustWaterMass,
    "b,WaterManConc,final": WaterManConc,

    "b,SurfCO2Mass,final": SurfCO2Mass,       
    "b,ManCO2MAss,final": ManCO2Mass,
    "b,CrustCO2Mass,final": CrustCO2Mass,

    "b,OxygenMass,final": OxygenMass
}

with open('Unconverged_Param_Dictionary.json', 'w') as f:
    json.dump(output, f)

ValidRows = []

# Get the number of rows based on the length of the lists in the dictionary
#print(repr(output["Age,final"]))
iNumRows = len(output["b,SurfWaterMass,final"])
for i in range(iNumRows):
    bValid=1
    for key in output:
        try:
            if np.isnan(output[key][i]):
                #print(repr(i)+": NaN for "+repr(key))
                bValid=0
        except:
            print(repr(key) + " " +repr(i))

    if StopTime[i] < Time[i]:
        bValid=0
        # Extract the row as a list of values
    # if output["b,CO2MassAtm,final"][i] < 0:
    #     #print("Final atm CO2 mass < 0") 
    #     bValid=0
    # if output["b,SurfWaterMass,final"][i] > 1e30 and bValid:
    #     print(repr(i)+": "+repr(output["b,SurfWaterMass,final"][i])+ " "+repr(SurfWaterMass[i]),flush=True)
    #     bValid = 0


    if bValid:
        output["Time,final"][i] /= YearSec
        output["StopTime"][i] /= YearSec
        #print(repr(output["Age,final"][i])+ " 1")
        output["Age,final"][i] /= YearSec
        #print(repr(output["Age,final"][i])+ " 2")

        #print(repr(output["b,SurfWaterMass,final"][i]),flush=True)
        output["b,SurfWaterMass,final"][i] /= -TO
        #print(repr(output["b,SurfWaterMass,final"][i]),flush=True)
        output["b,WaterMassSol,final"][i] *= -1
        #output["b,CO2MassSol,final"][i] /= -TO


        row = [output[key][i] for key in output]
        ValidRows.append(row)

# Convert the list of valid rows to a NumPy array
ValidArray = np.array(ValidRows)

# Save the NumPy array to a .npy file
np.save("MagmaOceanFinal.npy", ValidArray)
#print(f"Filtered data successfully saved to {file_path}")





print("Processed "+repr(found)+" directories.")
print(repr(missing)+" log files missing.")
print('Number of sims ready for stagnant lid: '+repr(len(ValidRows)))
#print(SurfTemp)
