import json
import pandas as pd
import numpy as np
from pathlib import Path

body = 'b'
vplanet_logfile = 'gj1132.log'

SurfTemp = []
WaterMassSol = []
WaterMassMOAtm = []
PressWaterAtm = []
SurfWaterMass = []
OxygenMass = []
PressOxygenAtm = []
FracFe2O3Man = []
CO2MassSol = []
CO2MassMOAtm = []
PressCO2Atm = []
CO2MassAtm = []
Age = []
StopTime = []
Time = []



def save_filtered_dict_to_npy(file_path, data_dict, key_to_check, min_value, max_value):
    """
    Save a dictionary to a .npy file line by line after filtering based on a value check.

    Parameters:
    - file_path (str): The path where the .npy file will be saved.
    - data_dict (dict): Dictionary where each key maps to a list of values.
    - key_to_check (str): The key in the dictionary whose values will be checked.
    - min_value (numeric): The minimum acceptable value for the check.
    - max_value (numeric): The maximum acceptable value for the check.

    Raises:
    - ValueError: If the key_to_check is not present in the dictionary.
    """
    # Check if the key_to_check is in the dictionary
    if key_to_check not in data_dict:
        raise ValueError(f"Key '{key_to_check}' not found in the dictionary.")
    
    # Initialize an empty list to store valid rows
    valid_rows = []

    # Get the number of rows based on the length of the lists in the dictionary
    num_rows = len(next(iter(data_dict.values())))

    # Iterate through each row index
    for i in range(num_rows):
        # Check if the current row meets the criteria
        if min_value <= data_dict[key_to_check][i] <= max_value:
            # Extract the row as a list of values
            row = [data_dict[key][i] for key in data_dict]
            valid_rows.append(row)

    # Convert the list of valid rows to a NumPy array
    valid_array = np.array(valid_rows)

    # Save the NumPy array to a .npy file
    np.save(file_path, valid_array)
    print(f"Filtered data successfully saved to {file_path}")




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
            
            if final and system and parts[0] == '(Age)':
                Age.append(float(parts[-1]))

            if final and parts[1] == 'BODY:' and parts[2] == body:
                ready = 1

            if ready and parts[0] == '(SurfTemp)':
                SurfTemp.append(float(parts[-1]))

            if ready and parts[0] == '(SurfWaterMass)':
                SurfWaterMass.append(float(parts[-1]))
            
            if ready and parts[0] == '(WaterMassSol)':
                WaterMassSol.append(float(parts[-1]))

            if ready and parts[0] == '(WaterMassMOAtm)':
                WaterMassMOAtm.append(float(parts[-1]))

            if ready and parts[0] == '(PressWaterAtm)':
                PressWaterAtm.append(float(parts[-1]))

            if ready and parts[0] == '(PressOxygenAtm)':
                PressOxygenAtm.append(float(parts[-1]))

            if ready and parts[0] == '(OxygenMass)':
                OxygenMass.append(float(parts[-1]))

            if ready and parts[0] == '(CO2MassSol)' and co2masssol == 0:
                CO2MassSol.append(float(parts[-1]))
                co2masssol = 1

            if ready and parts[0] == '(CO2MassSol)' and co2masssol == 1:
                CO2MassAtm.append(float(parts[-1]))  # Typo in vplanet! 2 CO2MassSol's!

            if ready and parts[0] == '(CO2MassMOAtm)':
                CO2MassMOAtm.append(float(parts[-1]))

            if ready and parts[0] == '(FracFe2O3Man)':
                FracFe2O3Man.append(float(parts[-1]))

            if ready and parts[0] == '(PressCO2Atm)':
                PressCO2Atm.append(float(parts[-1]))

            if ready and parts[0] == '(Time)':
                Time.append(float(parts[-1]))

        # print (SurfTemp[0])                
        # print (WaterMassSol[0])                
        # print (WaterMassMOAtm[0])                
        # print (PressWaterAtm[0])                
        # print (PressOxygenAtm[0])                
        # print (CO2MassSol[0])                
        # print (CO2MassMOAtm[0])
        # print (FracFe2O3Man[0])
        # print (PressCO2Atm[0])
                        
        # exit()

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

output = {
    "Time,final": Time,
    "StopTime": StopTime,
    "Age,final": Age,
    "b,SurfTemp,final": SurfTemp,

    "b,WaterMassSol,final": WaterMassSol,
    "b,WaterMassMOAtm,final": WaterMassMOAtm,
    "b,PressWaterAtm,final": PressWaterAtm,
    "b,SurfWaterMass,final": SurfWaterMass,

    "b,OxygenMass,final": OxygenMass,
    "b,PressOxygenAtm,final": OxygenMass,
    "b,FracFe2O3Man,final": FracFe2O3Man,

    "b,CO2MassSol,final": CO2MassSol,
    "b,CO2MassMOAtm,final": CO2MassMOAtm,
    "b,CO2MassAtm,final": CO2MassAtm,
    "b,PressCO2Atm,final": PressCO2Atm
}

with open('Unconverged_Param_Dictionary.json', 'w') as f:
    json.dump(output, f)

ValidRows = []

# Get the number of rows based on the length of the lists in the dictionary
iNumRows = len(next(iter(output.values())))

for i in range(iNumRows):
    # Check if the current row meets the criteria
    if StopTime[i] < Time[i]:

        # Extract the row as a list of values
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