import re
from matplotlib import pyplot as plt
import pandas as pd
import os
from cal_PlotAndFit_lookup import fit_mutiple, plot_and_fit_mutiple, plot_and_fit_mutipleWithRSME, plot_and_fit_overviewGreyAndRed

filepathToDirectory = "/home/yannick/Documents/SD/23-05-04_batcyc_preprocessing_result_6/prepr_res_eoc/"
togglePlotAllParameters = False
arrayParametersCellsToDisplay = [
    "P046",
]
arrayPackagesToDisplay = [
    "eoc",
]
arrayCalendarParameterID = ["P001", "P002", "P003", "P004", "P005", "P006",
                            "P007", "P008", "P009", "P010", "P011", "P012", "P013", "P014", "P005", "P016"]
arrayCyclicParameterID = ["P017", "P018", "P019", "P020", "P021", "P022", "P023", "P024", "P025", "P026", "P027", "P028", "P029", "P030", "P031", "P032", "P033", "P034", "P035", "P036", "P037", "P038", "P039",
                          "P040", "P041", "P042", "P043", "P044", "P045", "P046", "P047", "P048", "P048", "P049", "P050", "P051", "P052", "P053", "P054", "P055", "P056", "P057", "P058", "P059", "P060", "P061", "P062", "P063", "P064"]
arrayProfileParameterID = ["P065", "P066", "P067", "P068",
                           "P069", "P070", "P071", "P072", "P073", "P074", "P075", "P076"]

filesToSearch = arrayCalendarParameterID

patternExtractSlaveCellNumber = r'S(\d+)_C(\d+)\.csv'


files = os.listdir(filepathToDirectory)
files = [f for f in files if any(
    substring in f for substring in filesToSearch)]
print(files)
files.sort()
data = {
    'fileName': [],
    'filePath': [],
    'SlaveNumber': [],
    'CellNumber': [],
    'age_temp': [],
    'age_soc': []
}
df = pd.DataFrame(data)

for file in files:
    pattern = r'S(\d+)_C(\d+)\.csv'
    # Use the re.search() function to search for the pattern in the filename
    match = re.search(pattern, file)
    # Extract the S and C values from the match object
    s = int(match.group(1))
    c = int(match.group(2))

    columns_to_extract = ['age_temp', 'age_soc']
    # Read the specified columns from the CSV file
    temp_df = pd.read_csv(filepathToDirectory+file,
                          usecols=columns_to_extract, delimiter=';')
    age_temp = temp_df['age_temp'].iloc[0]
    age_soc = temp_df['age_soc'].iloc[0]
    file_data = {
        'fileName': file,
        'filePath': filepathToDirectory+file,
        'SlaveNumber': s,
        'CellNumber': c,
        'age_temp': age_temp,
        'age_soc': age_soc
    }
    df = pd.concat([df, pd.DataFrame(file_data, index=[0])], ignore_index=True)

print(df.head())
arrayPath = df['filePath'].to_list()

print(arrayPath)
# plot_and_fit_mutipleWithRSME(arrayPath)
plot_and_fit_overviewGreyAndRed(arrayPath)

# fit_mutiple(arrayPath)
