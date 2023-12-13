import os
import logging as lg
import re

import pandas as pd

# Set variables for utils
filepathToDirectory = "C:\\Users\\Yannick\\bwSyncShare\\MA Yannick Fritsch\\00_Daten\\prepr_res_eoc\\"

arrayCalendarParameterID = ["P001", "P002", "P003", "P004", "P005", "P006","P007", "P008", "P009", "P010", "P011", "P012", "P013", "P014", "P015", "P016"]
arrayCyclicParameterID = ["P017", "P018", "P019", "P020", "P021", "P022", "P023", "P024", "P025", "P026", "P027", "P028", "P029", "P030", "P031", "P032", "P033", "P034", "P035", "P036", "P037", "P038", "P039", "P040", "P041", "P042", "P043", "P044", "P045", "P046", "P047", "P048", "P048", "P049", "P050", "P051", "P052", "P053", "P054", "P055", "P056", "P057", "P058", "P059", "P060", "P061", "P062", "P063", "P064"]
arrayProfileParameterID = ["P065", "P066", "P067", "P068","P069", "P070", "P071", "P072", "P073", "P074", "P075", "P076"]


#   Erstellt ein Dataframe mit einer Liste von alle Dateien die im Ordner gefunden werden.
#   Für jede Datei werden 'fileName, filePath, SlaveNumber, CellNumber, age_temp, age_soc, parameterId
#   Dataframe beinhaltet noch keine Messdaten

def helper_df_selected_parameters(filesToSearch: list):
    files = os.listdir(filepathToDirectory)
    files = [f for f in files if any(substring in f for substring in filesToSearch)]
    lg.info(f"Found {len(files)} csv files")
    files.sort()
    data = {
        'fileName': [],
        'filePath': [],
        'SlaveNumber': [],
        'CellNumber': [],
        'age_temp': [],
        'age_soc': [],
        "parameterId" : []
    }
    df = pd.DataFrame(data)
    #Fill the dataframe with info of csv files
    for file in files:
        # Extract PXXX
        p_number = re.search(r'P(\d+)', file).group(1)

        # Extract SXX
        s_number = re.search(r'S(\d+)', file).group(1)

        # Extract CXX
        c_number = re.search(r'C(\d+)', file).group(1)

        columns_to_extract = ['age_temp', 'age_soc']
        # Read the specified columns from the CSV file
        temp_df = pd.read_csv(filepathToDirectory+file,usecols=columns_to_extract, delimiter=';', nrows=1)
        age_temp = temp_df['age_temp'].iloc[0]
        age_soc = temp_df['age_soc'].iloc[0]
        file_data = {
            'fileName': file,
            'filePath': filepathToDirectory+file,
            'SlaveNumber': s_number,
            'CellNumber': c_number,
            'age_temp': age_temp,
            'age_soc': age_soc,
            "parameterId" : p_number
        }
        df = pd.concat([df, pd.DataFrame(file_data, index=[0])], ignore_index=True)
    return df