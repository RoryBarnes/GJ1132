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
PressOxygenAtm = []
FracFe2O3Man = []
CO2MassSol = []
CO2MassMOAtm = []
PressCO2Atm = []
Age = []

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
            
            if ready and parts[0] == '(WaterMassSol)':
                WaterMassSol.append(float(parts[-1]))

            if ready and parts[0] == '(WaterMassMOAtm)':
                WaterMassMOAtm.append(float(parts[-1]))

            if ready and parts[0] == '(PressWaterAtm)':
                PressWaterAtm.append(float(parts[-1]))

            if ready and parts[0] == '(PressOxygenAtm)':
                PressOxygenAtm.append(float(parts[-1]))

            if ready and parts[0] == '(CO2MassSol)' and co2masssol == 0:
                CO2MassSol.append(float(parts[-1]))
                co2masssol = 1

            if ready and parts[0] == '(CO2MassMOAtm)':
                CO2MassMOAtm.append(float(parts[-1]))

            if ready and parts[0] == '(FracFe2O3Man)':
                FracFe2O3Man.append(float(parts[-1]))

            if ready and parts[0] == '(PressCO2Atm)':
                PressCO2Atm.append(float(parts[-1]))

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
    "Age,final": Age,
    "b,SurfTemp,final": SurfTemp,
    "b,WaterMassSol,final": WaterMassSol,
    "b,WaterMassMOAtm,final": WaterMassMOAtm,
    "b,PressWaterAtm,final": PressWaterAtm,
    "b,PressOxygenAtm,final": PressOxygenAtm,
    "b,CO2MassSol,final": CO2MassSol,
    "b,CO2MassMOAtm,final": CO2MassMOAtm,
    "b,FracFe2O3Man,final": FracFe2O3Man,
    "b,PressCO2Atm,final": PressCO2Atm
}

with open('Unconverged_Param_Dictionary.json', 'w') as f:
    json.dump(output, f)

print("Processed "+repr(found)+" directories.")
print(repr(missing)+" log files missing.")
#print(SurfTemp)