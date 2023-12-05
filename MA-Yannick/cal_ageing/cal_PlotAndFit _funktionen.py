import os
import re
import time

import pandas as pd
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import logging as lg
import seaborn as sns
import csv
# lg.basicConfig(level=lg.DEBUG)
np.set_printoptions(suppress=True)

# Global variables
max_capacity = 0
age_temp = 0
age_voltage = 0
exponent =1

# DEFINE PATHS TO CSV FILES
filepathToDirectory = "C:\\Users\\Yannick\\bwSyncShare\\MA Yannick Fritsch\\00_Daten\\prepr_res_eoc\\"
testPath = "/home/yannick/Documents/SD/23-05-04_batcyc_preprocessing_result_6/prepr_res_eoc/cell_eoc_P013_3_S19_C09.csv"
tempFilesDirectory = "C:\\Users\\Yannick\\Desktop\\tempSkripte\\"
saveFigsDirectory = "C:\\Users\\Yannick\\bwSyncShare\\MA Yannick Fritsch\\02_Arbeit\\Grafiken\\"
df_alleZellen = pd.DataFrame(columns=['nameCell', 'xdata', 'ydata','age_voltage','age_temp'])
uniqueZellen = []

## HELPER FUNCTION FOR PLOTTING
filepathToDirectory = "C:\\Users\\Yannick\\bwSyncShare\\MA Yannick Fritsch\\00_Daten\\prepr_res_eoc\\"

arrayPackagesToDisplay = ["eoc",]
arrayCalendarParameterID = ["P001", "P002", "P003", "P004", "P005", "P006","P007", "P008", "P009", "P010", "P011", "P012", "P013", "P014", "P015", "P016"]
arrayCyclicParameterID = ["P017", "P018", "P019", "P020", "P021", "P022", "P023", "P024", "P025", "P026", "P027", "P028", "P029", "P030", "P031", "P032", "P033", "P034", "P035", "P036", "P037", "P038", "P039", "P040", "P041", "P042", "P043", "P044", "P045", "P046", "P047", "P048", "P048", "P049", "P050", "P051", "P052", "P053", "P054", "P055", "P056", "P057", "P058", "P059", "P060", "P061", "P062", "P063", "P064"]
arrayProfileParameterID = ["P065", "P066", "P067", "P068","P069", "P070", "P071", "P072", "P073", "P074", "P075", "P076"]
ARRAY_CELLS_TO_DISPLAY = arrayCalendarParameterID


# Define the functions selected in theory part
    
def funktion1(x, a0, a1, a2):
    
    return 1 - a0 * np.exp(a1*age_voltage)*np.exp(a2/age_temp)*np.power(x, exponent)
def funktion1_expo(x, a0, a1, a2, a3):
    return 1 - a0 * np.exp(a1*age_voltage)*np.exp(a2/age_temp)*np.power(x, a3)

# def funktion2(x,a0,a1):
#     return 1-a0*np.exp(a1/age_voltage)*np.power(x,exponent)
# def funktion2_expo(x,a0,a1,a2):
    # return 1-a0*np.exp(a1/age_voltage)*np.power(x,a2)
def funktion2(x, a0, a1, a2, a3):
    return 1 - (a0 + a1*age_voltage + a2*(age_voltage*age_voltage)) * np.exp(a3/age_temp) * np.power(x, exponent)
def funktion2_expo(x, a0, a1, a2, a3,a4):
    return 1 - (a0 + a1*age_voltage + a2*(age_voltage*age_voltage)) * np.exp(a3/age_temp) * np.power(x, a4)
def funktion3(x, a0, a1, a2):
    return 1 - (a0*age_voltage+a1)*np.exp(a2/age_temp)*np.power(x,exponent)
def funktion3_expo(x, a0, a1, a2, a3):
    return 1 - (a0*age_voltage+a1)*np.exp(a2/age_temp)*np.power(x,a3)
# def funktion5(x,a0,a1,a2):
#     return 1- a0*np.exp(a1*age_voltage)*np.exp(a2*age_voltage/age_temp)*np.power(x,exponent)
# def funktion5_expo(x,a0,a1,a2,a3):
#     return 1- a0*np.exp(a1*age_voltage)*np.exp(a2*age_voltage/age_temp)*np.power(x,a3)
# def funktion6(x, a0, a1,a2,a3):
#     return 1-a0*np.exp(a1*age_voltage)*np.exp((a2*age_voltage+a3)/age_temp)*np.power(x,exponent)
# def funktion6_expo(x, a0, a1,a2,a3,a4):
#     return 1-a0*np.exp(a1*age_voltage)*np.exp((a2*age_voltage+a3)/age_temp)*np.power(x,a4)
# # -------------------------------------------------------------------------------------------------------------------------------------------

def funktionExponent(x,a0,a1,a2):
    return a0+a1*x+a2*x*x
def funktionExponentMitWerten(x):
    return 0.9272072696667285-0.0072208866623697365*x+4.6916412031700374e-05*x*x
def flatten_2d_array(arr):
    return np.array(arr).flatten()
# -------------------------------------------------------------------------------------------------------------------------------------------

#  Returns a dataframe of the requestet files
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
# -------------------------------------------------------------------------------------------------------------------------------------------
def plot_capacity_over_time(arrayOfPathsToCsvFiles: list):
    df_cells = pd.DataFrame()
    df_cell_csv_content = pd.DataFrame()


    lg.debug(f"Plotting capacity over time")
    if len(arrayOfPathsToCsvFiles) == 0:
        lg.error("No csv files to plot")
        return False
    if (len(arrayOfPathsToCsvFiles) == 1) and (arrayOfPathsToCsvFiles[0] == "all"):
        df_cells = helper_df_selected_parameters(ARRAY_CELLS_TO_DISPLAY)
        # arrayOfPathsToCsvFiles = helper_df_selected_parameters(arrayCalendarParameterID)['filePath'].to_list()
        lg.debug(arrayOfPathsToCsvFiles)
    else:
        df_cells = helper_df_selected_parameters(arrayOfPathsToCsvFiles)

    print(df_cells.head())
    
    # Plot the capacity over time with temperature as color

    #Open each csv file of selected cells and add the capacity to the plot
    for index,row in df_cells.iterrows():
        df = pd.read_csv(row["filePath"], sep=";", header=0)
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2) & (df['cyc_condition'] == 2)]
        df.loc[:, 'cap_aged_est_Ah'] = df['cap_aged_est_Ah'] / df['cap_aged_est_Ah'].iloc[0]
        df["fileName"] = row["fileName"]
        df_cell_csv_content = pd.concat([df_cell_csv_content,df],ignore_index=True)
    # print(df_cell_csv_content)
    colorPalette = {0.0:"blue",10: "green",25: "orange", 40: "red"}
    lineStyle = {10:":",50:"-.",90:"-", 100:"--"}

    uniqueFiles= df_cell_csv_content["fileName"].unique()

    for file in uniqueFiles:
        df = df_cell_csv_content[df_cell_csv_content["fileName"] == file].reset_index()
        # print(df.head())
        colorInt = df["age_temp"][0]
        lineStyleInt = df["age_soc"][0]

        sns.lineplot(data=df, x="timestamp_s", y = "cap_aged_est_Ah", color = colorPalette[colorInt]) #linestyle = lineStyle[lineStyleInt] )
    # sns.lineplot(data=df_cell_csv_content, x="timestamp_s", y = "cap_aged_est_Ah", hue="age_temp", style="age_soc",estimator='median' )
    plt.xlabel("Zeit (s)")
    plt.ylabel("Kapazität (Ah) beim Checkup")
    plt.title("Plot of Timestamp vs Cap")
    
    # Display the plot
    plt.show()

    uniqueSoc = df_cell_csv_content["age_soc"].unique()
    uniqueTemp= df_cell_csv_content["age_temp"].unique()
    maxTimestamp = int(df_cell_csv_content["timestamp_s"].max())
    minTimestamp = int(df_cell_csv_content["timestamp_s"].min())
    max_capacity = 1
    min_capacity = 0.92



# Jedes SOC durchgehen
    for soc in uniqueSoc:
        # Jede Temeperatur durchgehen
        fig, ax = plt.subplots(ncols=1, nrows=1)
        for temp in uniqueTemp:
            # Lade in tempor
            df = df_cell_csv_content[(df_cell_csv_content["age_soc"] == soc) & (df_cell_csv_content["age_temp"] == temp)].reset_index()
            #select cap_aged_est_Ah and timestamp_s from df
            df = df[['timestamp_s', 'cap_aged_est_Ah',"fileName"]]
            uniqueFiles = df["fileName"].unique()
            y_values =[]
            for uniqueFile in uniqueFiles:
                y_values.append(df[df["fileName"] == uniqueFile]["cap_aged_est_Ah"].values)
            min_values = []
            max_values = []
            mean_values = []
            # print(*y_values)

            for values in zip(*y_values):
                min_values.append(min(values))
                max_values.append(max(values))
                mean_values.append(sum(values) / len(values))
            # print(min_values)
            # print(max_values)
            print(f'--------soc: {soc}, temp: {temp}-----------')
            # print(mean_values)
            # CALCULATE RMSE
            # print("mean",np.mean(y_values, axis=0))
            # mean = np.mean(y_values, axis=0)
            # print("standard deviation",np.std(y_values, axis=0))
            # print("Mean of standard deviation",np.mean(np.std(y_values, axis=0), axis=0))
            # print("rsme",np.sqrt(np.mean((mean-y_values)**2)))
                



            
            # x_values = np.array([i for i in range(0,(maxTimestamp-minTimestamp), int((maxTimestamp-minTimestamp)/(len(min_values))))])
            x_values = np.array(np.linspace(0, (maxTimestamp-minTimestamp), len(min_values), dtype=int))
            # print(x_values)
            x_values = x_values/(60*60*24)
            # print(len(min_values),len(max_values),len(mean_values),len(x_values))
            print(f"Gradient : {(mean_values[-1]-mean_values[0])/(x_values[-1]-x_values[0])}")
            ax.plot(x_values, mean_values,color=colorPalette[temp], label=f'{temp}°C')
            ax.fill_between(x_values, min_values, max_values, interpolate = True, alpha = 0.2, color=colorPalette[temp])

            filename = f"avg_SOC_{soc}_T_{temp}.csv"
            data = list(zip(x_values, mean_values))
            with open(tempFilesDirectory+filename, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(["xvalues", "mean_values"])  # Write the header
                writer.writerows(data)  # Write the data rows

        plt.title(f"Kapzität im Verlauf der Zeit - Lagerung bei SOC: {soc}%")
        plt.xlabel('Alterung (Tage)')
        plt.ylabel('Relative Kapazität')
        ax.grid(True,'major',linewidth=0.3)
        ax.grid(True,'minor',linewidth=0.2)
        
        plt.ylim(min_capacity, max_capacity)
        plt.legend()
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        plt.show()
    
    # Plot 3D 
# -------------------------------------------------------------------------------------------------------------------------------------------
def plot_and_fit(pathToCsvFile: str):
    global age_temp, age_voltage, max_capacity
    df = pd.read_csv(pathToCsvFile, header=0, delimiter=";")

    df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
    age_temp = df['age_temp'].iloc[0]+273.15
    age_soc = df['age_soc'].iloc[0]
    age_voltage = age_soc * 4.2 / 100
    age_temp = float(age_temp)

    relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']]
    relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
    relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
    max_capacity = df['cap_aged_est_Ah'].iloc[0]
    relevant_df = relevant_df.reset_index()

    # Load the data from the pandas dataframe
    xdata = relevant_df['timestamp_s'].values
    ydata = relevant_df['cap_aged_est_Ah'].values

    # Fit the function to the data

    p0 = [1,1,1,1,1]
    popt, pcov = optimize.curve_fit(funktion1_expo, xdata, ydata, p0=p0)
    print(popt)
    experimentalYdata = ydata
    modelYdata =funktion1_expo(xdata, *popt)
    print("Measured yData ", experimentalYdata)
    print("Calculated yData",modelYdata )
    #  Calculate the MSE 
    mse = np.square(np.subtract(modelYdata, experimentalYdata)).mean()
    print("MSE", mse)
    # Plot the optimized and experimental data
    plt.plot(xdata, experimentalYdata, 'b-', label='experimental data')
    plt.plot(xdata, modelYdata, 'r-', label='optimized data')
    
    
    plt.xlabel('timestamp_s')
    plt.ylabel('cap_aged_est_Ah')
    plt.legend()
    plt.show()
# -------------------------------------------------------------------------------------------------------------------------------------------
def fit_time_exponent(arrayOfPathsToCsvFiles: list):
    global age_temp, age_voltage, max_capacity
    df_cells = pd.DataFrame()
    df_cell_csv_content = pd.DataFrame()
    df_cells_exponent = pd.DataFrame()


    lg.debug(f"Plotting capacity over time")
    if len(arrayOfPathsToCsvFiles) == 0:
        lg.error("No csv files to plot")
        return False
    if (len(arrayOfPathsToCsvFiles) == 1) and (arrayOfPathsToCsvFiles[0] == "all"):
        df_cells = helper_df_selected_parameters(ARRAY_CELLS_TO_DISPLAY)
        # arrayOfPathsToCsvFiles = helper_df_selected_parameters(arrayCalendarParameterID)['filePath'].to_list()
        lg.debug(arrayOfPathsToCsvFiles)
    else:
        df_cells = helper_df_selected_parameters(arrayOfPathsToCsvFiles)

    print(df_cells.head())
    for index,row in df_cells.iterrows():
        df = pd.read_csv(row["filePath"], sep=";", header=0)
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2) & (df['cyc_condition'] == 2)]
        df.loc[:, 'cap_aged_est_Ah'] = df['cap_aged_est_Ah'] / df['cap_aged_est_Ah'].iloc[0]
        df["fileName"] = row["fileName"]
        df_cell_csv_content = pd.concat([df_cell_csv_content,df],ignore_index=True)
    print(df_cell_csv_content.head())
    uniqueFiles= df_cell_csv_content["fileName"].unique()

    for file in uniqueFiles:
        df = df_cell_csv_content[df_cell_csv_content["fileName"] == file].reset_index()
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        df = df[['timestamp_s', 'cap_aged_est_Ah']]
        df.loc[:, 'timestamp_s'] = df['timestamp_s'] - df['timestamp_s'].iloc[0]
        df.loc[:, 'cap_aged_est_Ah'] = df['cap_aged_est_Ah'] / df['cap_aged_est_Ah'].iloc[0]
        df = df.reset_index()
        # Load the data from the pandas dataframe
        xdata = df['timestamp_s'].values
        ydata = df['cap_aged_est_Ah'].values
        print(xdata,ydata)

        # Fit the FUNKTION1 to the data and extract the time exponent
        p0 = [1,1,1,0.6]
        popt, pcov = optimize.curve_fit(funktion1_expo, xdata, ydata, p0=p0)
        temp_df = pd.DataFrame({"fileName":file, "fitFunction": "funktion1", "exponent":popt[3], "age_temp": age_temp, "age_soc":age_soc}, index=[0])
        df_cells_exponent = pd.concat([df_cells_exponent,temp_df],ignore_index=True)
        
         # Fit the FUNKTION2 to the data and extract the time exponent
        p0 = [1,1,0.6]
        popt, pcov = optimize.curve_fit(funktion2_expo, xdata, ydata, p0=p0)
        temp_df = pd.DataFrame({"fileName":file, "fitFunction": "funktion2", "exponent":popt[2], "age_temp": age_temp, "age_soc":age_soc}, index=[0])
        df_cells_exponent = pd.concat([df_cells_exponent,temp_df],ignore_index=True)
         # Fit the FUNKTION3 to the data and extract the time exponent
        p0 = [1,1,1,1,0.6]
        popt, pcov = optimize.curve_fit(funktion3_expo, xdata, ydata, p0=p0)
        temp_df = pd.DataFrame({"fileName":file, "fitFunction": "funktion3", "exponent":popt[4], "age_temp": age_temp, "age_soc":age_soc}, index=[0])
        df_cells_exponent = pd.concat([df_cells_exponent,temp_df],ignore_index=True)
         # Fit the FUNKTION4 to the data and extract the time exponent
        p0 = [1,1,1,0.6]
        popt, pcov = optimize.curve_fit(funktion4_expo, xdata, ydata, p0=p0)
        temp_df = pd.DataFrame({"fileName":file, "fitFunction": "funktion4", "exponent":popt[3], "age_temp": age_temp, "age_soc":age_soc}, index=[0])
        df_cells_exponent = pd.concat([df_cells_exponent,temp_df],ignore_index=True)
         # Fit the FUNKTION5 to the data and extract the time exponent
        p0 = [1,1,1,0.6]
        popt, pcov = optimize.curve_fit(funktion5_expo, xdata, ydata, p0=p0)
        temp_df = pd.DataFrame({"fileName":file, "fitFunction": "funktion5", "exponent":popt[3], "age_temp": age_temp, "age_soc":age_soc}, index=[0])
        df_cells_exponent = pd.concat([df_cells_exponent,temp_df],ignore_index=True)
         # Fit the FUNKTION6 to the data and extract the time exponent
        p0 = [1,1,1,1,0.6]
        popt, pcov = optimize.curve_fit(funktion6_expo, xdata, ydata, p0=p0)
        temp_df = pd.DataFrame({"fileName":file, "fitFunction": "funktion6", "exponent":popt[4], "age_temp": age_temp, "age_soc":age_soc}, index=[0])
        df_cells_exponent = pd.concat([df_cells_exponent,temp_df],ignore_index=True)
        print(df_cells_exponent.head())
        print("Calulated time exponent for", file)
    df_forboxplot = df_cells_exponent[(df_cells_exponent["exponent"]>0.5)]
    averages =df_forboxplot.groupby(['age_soc'])["exponent"].mean()
    print(averages.index.values, type(averages.index.values))
    print(averages.values, type(averages.values))
    xdata=averages.index.values.astype(float)
    ydata=averages.values.astype(float)
    p0 = [1,1,1]
    popt, pcov = optimize.curve_fit(funktionExponent, xdata, ydata, p0=p0)
    x_data2 = np.linspace(xdata[0], xdata[-1], 50)
    print(x_data2, type(x_data2))
    print(f"Modell für Exponent: {popt[0]}+{popt[1]}*x+{popt[2]}*x^2")
    sns.boxplot(data=df_forboxplot, x="age_soc", y = "exponent", width = 0.5) 
    plt.show()
    sns.lineplot(x = x_data2,y = funktionExponent(x_data2,*popt))
    plt.show()

# -------------------------------------------------------------------------------------------------------------------------------------------
# 
def plot_and_fit_mutiple(arrayOfPathToCsvFiles: np.array):
    global age_temp, age_voltage
    # show the right number of columns
    col_number = 3 if len(arrayOfPathToCsvFiles) > 3 else len(
        arrayOfPathToCsvFiles)
    row_number = int(len(arrayOfPathToCsvFiles))//3 + \
        1 if len(arrayOfPathToCsvFiles) > 3 else 1
    fig, subfigs = plt.subplots(nrows=row_number, ncols=col_number)
    fig.suptitle('Calendar ageing Cells, experimental vs fitting')
    flatSubfigs = flatten_2d_array(subfigs)
    for pathToCell, subfigure in zip(arrayOfPathToCsvFiles, flatSubfigs):
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()

        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        ydata = relevant_df['cap_aged_est_Ah'].values

        # p0 = [0.2, 0.1, -0.027, -20, 0.7]
        # lower_bounds = [-100, -100, -1, -22, 0.5]
        # upper_bounds = [100, 100, 0, -18, 0.8]

        # popt, pcov = optimize.curve_fit(funktion1, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))

        # p0 = [7,1,1,10]
        # lower_bounds = [-50, -50, -5, -15]
        # upper_bounds = [100, 50, 5, 15] #func 2
        p0 = [1,1,1]
        lower_bounds = [-1, -50, 0.5]
        upper_bounds = [1, 50, 1] #func 3
        popt, pcov = optimize.curve_fit(func3, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
        print("pot",popt)
        subfigure.plot(xdata, ydata, 'b-', label='experimental data')
        # print("xdata", xdata, "*popt", *popt)
        subfigure.plot(xdata, func3(xdata, *popt), 'r-', label='fitting')
        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"

        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            subfigure.set_title(extracted_value)
        # subfigure.set_xlabel('timestamp_s')
        # subfigure.set_ylabel('cap_aged_est_Ah')
    if len(arrayOfPathToCsvFiles) > 1:
        # Remove empty subplots
        for ax in subfigs.flat:
            if not ax.lines:
                ax.remove()
    # Adjust the layout
    # plt.tight_layout()
    # Set the title of the axis
    plt.xlabel('timestamp_s')
    plt.ylabel('cap_aged_est_Ah')
    plt.legend()
    plt.show()




def fit_mutiple(arrayOfPathToCsvFiles):
    global age_temp, age_voltage, max_capacity, exponent, df_alleZellen
    df_popt = pd.DataFrame(columns=['nameCell', 'a0', 'a1', 'a2', 'a3'])
    fig, ax = plt.subplots()

    for pathToCell in arrayOfPathToCsvFiles:
        # print(pathToCell)
        # Read CSV file
        
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        # Select checkup data
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        # Select the ageing temperature and the ageing SOC with the first row of the select rows
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        exponent=0.5
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()

        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()


        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        ydata = relevant_df['cap_aged_est_Ah'].values
        
        
        p0 = [4,2,3]
        lower_bounds = [-5, -5,  -1000]
        upper_bounds = [5, 5, 100]

        popt, pcov = optimize.curve_fit(funktion1, xdata, ydata, p0=p0, bounds=[lower_bounds, upper_bounds])
        print(popt)
        ax.plot(xdata, ydata, 'b-', label='experimental data')
        # print("xdata", xdata, "*popt", *popt)
        ax.plot(xdata, funktion1(xdata, *popt), 'r-', label='fitting')
        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            new_row = {'nameCell': extracted_value,
                       'a0': popt[0], 'a1': popt[1], 'a2': popt[2]}
        else:
            new_row = {'nameCell': "unknown",
                       'a0': popt[0], 'a1': popt[1], 'a2': popt[2], 'a3': popt[3]}
        new_df = pd.DataFrame(new_row, index=[0])
        # 
        df_popt = pd.concat([df_popt, new_df], ignore_index=True)
    plt.show()
    print(df_popt.head())
    # print(df_popt.drop("nameCell").mean().values)

    # Calculate the RMSE
   # rmse = np.sqrt(np.mean((y_noisy - y_pred) ** 2))



def funktion1_min(parameters):
    global uniqueZellen,df_alleZellen,age_voltage,age_temp
    # Für jede Zelle
    a0= parameters[0]
    a1= parameters[1]
    a2= parameters[2]
    # a3= parameters[3] 
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
        age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
        
        y_modellierte= funktion1(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors

def funktion1_minimize_fixed(arrayOfPathToCsvFiles):

    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Calendar\cal_funktion1_fixed.csv"
    global age_temp, age_voltage,maxXValue,exponent,uniqueZellen,df_alleZellen
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cal_funktion1_00C_fixed")
    fig2 = plt.figure("cal_funktion1_10C_fixed")
    fig3 = plt.figure("cal_funktion1_25C_fixed")
    fig4 = plt.figure("cal_funktion1_40C_fixed")
    temperatureFig = {273.15:fig1 ,283.15: fig2 ,298.15: fig3, 313.15: fig4}

#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0
    
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        exponent = 0.5

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        
        if age_temp==273.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==283.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==298.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==313.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'age_voltage':age_voltage, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()

    start = [1,1,1]
    bounds = [(-5,5),(0,1),(-1000,1000)]
    startTime = time.time()
    print(f"Starting optimization {startTime}")
    result = optimize.minimize(funktion1_min,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [0.03099997256512114, 0.13747368552595407, -999.999998685382, 0.6894569540598147]
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    
    temp = [273.15, 283.15, 298.15, 313.15]
    volt= [0.42, 2.1, 3.78, 4.2]
    errorList = []
    globalErrorList = []
    for i_t in range(0, len(temp)):
        errorList = []
        age_temp = temp[i_t]
        for i_v in range(0, len(volt)):
            age_voltage = volt[i_v]
            

            sns.lineplot(x = xdata, y = funktion1(xdata, *result.x), color = colorPalette[age_temp], ax = temperatureFig[age_temp].gca(), linestyle = lineStyle[age_voltage])

            # Calculate error
            for cell in uniqueZellen:
                if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0] == age_voltage) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0] == age_temp):
                    xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                    ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                    age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
                    age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
                    
                    y_modellierte= funktion1(xdata, *result.x)
                    error = (y_modellierte-ydata)**2
                    errorList.append(error*1000)
                    globalErrorList.append(error*1000)
            rsmeSOC  = np.sqrt(np.mean(errorList))
            print(f"RSME SOC age_voltage: {rsmeSOC}")
        rmseTemperature = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=0, label ="SOC")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle=':', label ="10%")
        line3 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='--', label ="50%")
        line4 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-.', label ="90%")
        line5 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-',label ="100%")
        legend1 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = 0.5 #"%.3f" % result.x[3]
        rmseString = "%.3f" % rmseTemperature
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        temperatureFig[age_temp].gca().add_artist(legend1)
        temperatureFig[age_temp].gca().add_artist(legend2)


        temperatureFig[age_temp].gca().grid(color='lightgrey', linestyle='--')
        temperatureFig[age_temp].gca().set_xlabel(f'Zeit (Tage)')
        temperatureFig[age_temp].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        temperatureFig[age_temp].gca().set_title(f'Kalendarische Modellierung - Funktion Nr. 1 bei: {age_temp-273.15}°C')




    rsme = "%.4f" % np.sqrt(np.mean(globalErrorList))
    print(f"Root Mean Squared Error: {rsme}")
    optimal_parameters = result.x
    optimal_function_value = result.fun

    csv_filename = fileToSaveValues
    header = ['Parameter {}'.format(i) for i in range(len(optimal_parameters))] + ['Optimal Function Value'] +['RSME']
    data = [np.array(optimal_parameters).tolist() + [optimal_function_value] + [np.sqrt(np.mean(globalErrorList))]]

    # Write the data to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile,delimiter =';')
        csv_writer.writerow(header)
        csv_writer.writerows(data)

    fig1.savefig(saveFigsDirectory+"cal_funktion1_00C_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cal_funktion1_10C_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cal_funktion1_25C_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cal_funktion1_40C_fixed.png", format='png',  pad_inches=0, transparent=False)


    plt.show()

def funktion1_min_expo(parameters):
    global uniqueZellen,df_alleZellen,age_voltage,age_temp
    # Für jede Zelle
    a0= parameters[0]
    a1= parameters[1]
    a2= parameters[2]
    # a3= parameters[3] 
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
        age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
        
        y_modellierte= funktion1_expo(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors

def funktion1_minimize_expo(arrayOfPathToCsvFiles):

    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Calendar\cal_funktion1_expo.csv"
    global age_temp, age_voltage,maxXValue,exponent,uniqueZellen,df_alleZellen
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cal_funktion1_00C_expo")
    fig2 = plt.figure("cal_funktion1_10C_expo")
    fig3 = plt.figure("cal_funktion1_25C_expo")
    fig4 = plt.figure("cal_funktion1_40C_expo")
    temperatureFig = {273.15:fig1 ,283.15: fig2 ,298.15: fig3, 313.15: fig4}

#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0
    
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        exponent = 0.5

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        
        if age_temp==273.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==283.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==298.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==313.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'age_voltage':age_voltage, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()

    start = [1,1,1,0.6]
    bounds = [(-5,5),(0,1),(-1000,1000),(0.2,0.9)]
    startTime = time.time()
    print(f"Starting optimization {startTime}")
    result = optimize.minimize(funktion1_min_expo,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [0.03099997256512114, 0.13747368552595407, -999.999998685382, 0.6894569540598147]
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    
    temp = [273.15, 283.15, 298.15, 313.15]
    volt= [0.42, 2.1, 3.78, 4.2]
    errorList = []
    globalErrorList = []
    for i_t in range(0, len(temp)):
        errorList = []
        age_temp = temp[i_t]
        for i_v in range(0, len(volt)):
            age_voltage = volt[i_v]
            

            sns.lineplot(x = xdata, y = funktion1_expo(xdata, *result.x), color = colorPalette[age_temp], ax = temperatureFig[age_temp].gca(), linestyle = lineStyle[age_voltage])

            # Calculate error
            for cell in uniqueZellen:
                if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0] == age_voltage) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0] == age_temp):
                    xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                    ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                    age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
                    age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
                    
                    y_modellierte= funktion1_expo(xdata, *result.x)
                    error = (y_modellierte-ydata)**2
                    errorList.append(error*1000)
                    globalErrorList.append(error*1000)
            rsmeSOC  = np.sqrt(np.mean(errorList))
            print(f"RSME SOC age_voltage: {rsmeSOC}")
        rmseTemperature = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=0, label ="SOC")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle=':', label ="10%")
        line3 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='--', label ="50%")
        line4 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-.', label ="90%")
        line5 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-',label ="100%")
        legend1 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = "%.3f" % result.x[3]
        rmseString = "%.3f" % rmseTemperature
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        temperatureFig[age_temp].gca().add_artist(legend1)
        temperatureFig[age_temp].gca().add_artist(legend2)


        temperatureFig[age_temp].gca().grid(color='lightgrey', linestyle='--')
        temperatureFig[age_temp].gca().set_xlabel(f'Zeit (Tage)')
        temperatureFig[age_temp].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        temperatureFig[age_temp].gca().set_title(f'Kalendarische Modellierung - Funktion Nr. 1 bei: {age_temp-273.15}°C')




    rsme = "%.4f" % np.sqrt(np.mean(globalErrorList))
    print(f"Root Mean Squared Error: {rsme}")
    optimal_parameters = result.x
    optimal_function_value = result.fun

    csv_filename = fileToSaveValues
    header = ['Parameter {}'.format(i) for i in range(len(optimal_parameters))] + ['Optimal Function Value'] +['RSME']
    data = [np.array(optimal_parameters).tolist() + [optimal_function_value] + [np.sqrt(np.mean(globalErrorList))]]

    # Write the data to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile,delimiter =';')
        csv_writer.writerow(header)
        csv_writer.writerows(data)

    fig1.savefig(saveFigsDirectory+ "cal_funktion1_00C_expo.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cal_funktion1_10C_expo.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cal_funktion1_25C_expo.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cal_funktion1_40C_expo.png", format='png',  pad_inches=0, transparent=False)


    plt.show()



def funktion2_min(parameters):
    global uniqueZellen,df_alleZellen,age_voltage,age_temp
    # Für jede Zelle
    a0= parameters[0]
    a1= parameters[1]
    a2= parameters[2]
    # a3= parameters[3] 
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
        age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
        
        y_modellierte= funktion2(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors

def funktion2_minimize_fixed(arrayOfPathToCsvFiles):

    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Calendar\cal_funktion2_fixed.csv"
    global age_temp, age_voltage,maxXValue,exponent,uniqueZellen,df_alleZellen
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cal_funktion2_00C_fixed")
    fig2 = plt.figure("cal_funktion2_10C_fixed")
    fig3 = plt.figure("cal_funktion2_25C_fixed")
    fig4 = plt.figure("cal_funktion2_40C_fixed")
    temperatureFig = {273.15:fig1 ,283.15: fig2 ,298.15: fig3, 313.15: fig4}

#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0
    
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        exponent = 0.5

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        
        if age_temp==273.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==283.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==298.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==313.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'age_voltage':age_voltage, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()

    start = [1,1,1,1]
    bounds = [(-5,5),(-5,5),(-5,5),(-1000,1000)]
    startTime = time.time()
    print(f"Starting optimization {startTime}")
    result = optimize.minimize(funktion2_min,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [0.03099997256512114, 0.13747368552595407, -999.999998685382, 0.6894569540598147]
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    
    temp = [273.15, 283.15, 298.15, 313.15]
    volt= [0.42, 2.1, 3.78, 4.2]
    errorList = []
    globalErrorList = []
    for i_t in range(0, len(temp)):
        errorList = []
        age_temp = temp[i_t]
        for i_v in range(0, len(volt)):
            age_voltage = volt[i_v]
            

            sns.lineplot(x = xdata, y = funktion2(xdata, *result.x), color = colorPalette[age_temp], ax = temperatureFig[age_temp].gca(), linestyle = lineStyle[age_voltage])

            # Calculate error
            for cell in uniqueZellen:
                if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0] == age_voltage) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0] == age_temp):
                    xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                    ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                    age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
                    age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
                    
                    y_modellierte= funktion2(xdata, *result.x)
                    error = (y_modellierte-ydata)**2
                    errorList.append(error*1000)
                    globalErrorList.append(error*1000)
            rsmeSOC  = np.sqrt(np.mean(errorList))
            print(f"RSME SOC age_voltage: {rsmeSOC}")
        rmseTemperature = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=0, label ="SOC")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle=':', label ="10%")
        line3 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='--', label ="50%")
        line4 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-.', label ="90%")
        line5 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-',label ="100%")
        legend1 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = 0.5 #"%.3f" % result.x[3]
        rmseString = "%.3f" % rmseTemperature
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        temperatureFig[age_temp].gca().add_artist(legend1)
        temperatureFig[age_temp].gca().add_artist(legend2)


        temperatureFig[age_temp].gca().grid(color='lightgrey', linestyle='--')
        temperatureFig[age_temp].gca().set_xlabel(f'Zeit (Tage)')
        temperatureFig[age_temp].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        temperatureFig[age_temp].gca().set_title(f'Kalendarische Modellierung - Funktion Nr. 2 bei: {age_temp-273.15}°C')




    rsme = "%.4f" % np.sqrt(np.mean(globalErrorList))
    print(f"Root Mean Squared Error: {rsme}")
    optimal_parameters = result.x
    optimal_function_value = result.fun

    csv_filename = fileToSaveValues
    header = ['Parameter {}'.format(i) for i in range(len(optimal_parameters))] + ['Optimal Function Value'] +['RSME']
    data = [np.array(optimal_parameters).tolist() + [optimal_function_value] + [np.sqrt(np.mean(globalErrorList))]]

    # Write the data to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile,delimiter =';')
        csv_writer.writerow(header)
        csv_writer.writerows(data)

    fig1.savefig(saveFigsDirectory+ "cal_funktion2_00C_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cal_funktion2_10C_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cal_funktion2_25C_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cal_funktion2_40C_fixed.png", format='png',  pad_inches=0, transparent=False)

    plt.show()

def funktion2_min_expo(parameters):
    global uniqueZellen,df_alleZellen,age_voltage,age_temp
    # Für jede Zelle
    a0= parameters[0]
    a1= parameters[1]
    a2= parameters[2]
    # a3= parameters[3] 
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
        age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
        
        y_modellierte= funktion2_expo(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors

def funktion2_minimize_expo(arrayOfPathToCsvFiles):

    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Calendar\cal_funktion2_expo.csv"
    global age_temp, age_voltage,maxXValue,exponent,uniqueZellen,df_alleZellen
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cal_funktion2_00C_expo")
    fig2 = plt.figure("cal_funktion2_10C_expo")
    fig3 = plt.figure("cal_funktion2_25C_expo")
    fig4 = plt.figure("cal_funktion2_40C_expo")
    temperatureFig = {273.15:fig1 ,283.15: fig2 ,298.15: fig3, 313.15: fig4}

#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0
    
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        exponent = 0.5

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        
        if age_temp==273.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==283.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==298.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==313.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'age_voltage':age_voltage, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()

    start = [1,1,1,1,0.6]
    bounds = [(-5,5),(-5,5),(0,1),(-1000,1000),(0.2,0.9)]
    startTime = time.time()
    print(f"Starting optimization {startTime}")
    result = optimize.minimize(funktion2_min_expo,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [0.03099997256512114, 0.13747368552595407, -999.999998685382, 0.6894569540598147]
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    
    temp = [273.15, 283.15, 298.15, 313.15]
    volt= [0.42, 2.1, 3.78, 4.2]
    errorList = []
    globalErrorList = []
    for i_t in range(0, len(temp)):
        errorList = []
        age_temp = temp[i_t]
        for i_v in range(0, len(volt)):
            age_voltage = volt[i_v]
            

            sns.lineplot(x = xdata, y = funktion2_expo(xdata, *result.x), color = colorPalette[age_temp], ax = temperatureFig[age_temp].gca(), linestyle = lineStyle[age_voltage])

            # Calculate error
            for cell in uniqueZellen:
                if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0] == age_voltage) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0] == age_temp):
                    xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                    ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                    age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
                    age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
                    
                    y_modellierte= funktion2_expo(xdata, *result.x)
                    error = (y_modellierte-ydata)**2
                    errorList.append(error*1000)
                    globalErrorList.append(error*1000)
            rsmeSOC  = np.sqrt(np.mean(errorList))
            print(f"RSME SOC age_voltage: {rsmeSOC}")
        rmseTemperature = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=0, label ="SOC")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle=':', label ="10%")
        line3 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='--', label ="50%")
        line4 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-.', label ="90%")
        line5 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-',label ="100%")
        legend1 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = "%.3f" % result.x[4]
        rmseString = "%.3f" % rmseTemperature
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        temperatureFig[age_temp].gca().add_artist(legend1)
        temperatureFig[age_temp].gca().add_artist(legend2)


        temperatureFig[age_temp].gca().grid(color='lightgrey', linestyle='--')
        temperatureFig[age_temp].gca().set_xlabel(f'Zeit (Tage)')
        temperatureFig[age_temp].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        temperatureFig[age_temp].gca().set_title(f'Kalendarische Modellierung - Funktion Nr. 2 bei: {age_temp-273.15}°C')




    rsme = "%.4f" % np.sqrt(np.mean(globalErrorList))
    print(f"Root Mean Squared Error: {rsme}")
    optimal_parameters = result.x
    optimal_function_value = result.fun

    csv_filename = fileToSaveValues
    header = ['Parameter {}'.format(i) for i in range(len(optimal_parameters))] + ['Optimal Function Value'] +['RSME']
    data = [np.array(optimal_parameters).tolist() + [optimal_function_value] + [np.sqrt(np.mean(globalErrorList))]]

    # Write the data to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile,delimiter =';')
        csv_writer.writerow(header)
        csv_writer.writerows(data)

    fig1.savefig(saveFigsDirectory+"cal_funktion2_00C_expo.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cal_funktion2_10C_expo.png", format='png', pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cal_funktion2_25C_expo.png", format='png', pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cal_funktion2_40C_expo.png", format='png', pad_inches=0, transparent=False)


    plt.show()



def funktion3_min(parameters):
    global uniqueZellen,df_alleZellen,age_voltage,age_temp
    # Für jede Zelle
    a0= parameters[0]
    a1= parameters[1]
    a2= parameters[2]
    # a3= parameters[3] 
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
        age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
        
        y_modellierte= funktion3(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors

def funktion3_minimize_fixed(arrayOfPathToCsvFiles):

    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Calendar\cal_funktion3_fixed.csv"
    global age_temp, age_voltage,maxXValue,exponent,uniqueZellen,df_alleZellen
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cal_funktion3_00C_fixed")
    fig2 = plt.figure("cal_funktion3_10C_fixed")
    fig3 = plt.figure("cal_funktion3_25C_fixed")
    fig4 = plt.figure("cal_funktion3_40C_fixed")
    temperatureFig = {273.15:fig1 ,283.15: fig2 ,298.15: fig3, 313.15: fig4}

#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0
    
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        exponent = 0.5

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        
        if age_temp==273.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==283.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==298.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==313.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'age_voltage':age_voltage, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()

    start = [1,1,1]
    bounds = [(-5,5),(-5,5),(-1000,1000)]
    startTime = time.time()
    print(f"Starting optimization {startTime}")
    result = optimize.minimize(funktion3_min,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [0.03099997256512114, 0.13747368552595407, -999.999998685382, 0.6894569540598147]
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    
    temp = [273.15, 283.15, 298.15, 313.15]
    volt= [0.42, 2.1, 3.78, 4.2]
    errorList = []
    globalErrorList = []
    for i_t in range(0, len(temp)):
        errorList = []
        age_temp = temp[i_t]
        for i_v in range(0, len(volt)):
            age_voltage = volt[i_v]
            

            sns.lineplot(x = xdata, y = funktion3(xdata, *result.x), color = colorPalette[age_temp], ax = temperatureFig[age_temp].gca(), linestyle = lineStyle[age_voltage])

            # Calculate error
            for cell in uniqueZellen:
                if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0] == age_voltage) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0] == age_temp):
                    xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                    ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                    age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
                    age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
                    
                    y_modellierte= funktion3(xdata, *result.x)
                    error = (y_modellierte-ydata)**2
                    errorList.append(error*1000)
                    globalErrorList.append(error*1000)
            rsmeSOC  = np.sqrt(np.mean(errorList))
            print(f"RSME SOC age_voltage: {rsmeSOC}")
        rmseTemperature = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=0, label ="SOC")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle=':', label ="10%")
        line3 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='--', label ="50%")
        line4 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-.', label ="90%")
        line5 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-',label ="100%")
        legend1 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = 0.5 #"%.3f" % result.x[3]
        rmseString = "%.3f" % rmseTemperature
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        temperatureFig[age_temp].gca().add_artist(legend1)
        temperatureFig[age_temp].gca().add_artist(legend2)


        temperatureFig[age_temp].gca().grid(color='lightgrey', linestyle='--')
        temperatureFig[age_temp].gca().set_xlabel(f'Zeit (Tage)')
        temperatureFig[age_temp].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        temperatureFig[age_temp].gca().set_title(f'Kalendarische Modellierung - Funktion Nr. 3 bei: {age_temp-273.15}°C')




    rsme = "%.4f" % np.sqrt(np.mean(globalErrorList))
    print(f"Root Mean Squared Error: {rsme}")
    optimal_parameters = result.x
    optimal_function_value = result.fun

    csv_filename = fileToSaveValues
    header = ['Parameter {}'.format(i) for i in range(len(optimal_parameters))] + ['Optimal Function Value'] +['RSME']
    data = [np.array(optimal_parameters).tolist() + [optimal_function_value] + [np.sqrt(np.mean(globalErrorList))]]

    # Write the data to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile,delimiter =';')
        csv_writer.writerow(header)
        csv_writer.writerows(data)

    fig1.savefig(saveFigsDirectory+"cal_funktion3_00C_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cal_funktion3_10C_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cal_funktion3_25C_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cal_funktion3_40C_fixed.png", format='png',  pad_inches=0, transparent=False)


    plt.show()

def funktion3_min_expo(parameters):
    global uniqueZellen,df_alleZellen,age_voltage,age_temp
    # Für jede Zelle
    a0= parameters[0]
    a1= parameters[1]
    a2= parameters[2]
    # a3= parameters[3] 
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
        age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
        
        y_modellierte= funktion3_expo(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors

def funktion3_minimize_expo(arrayOfPathToCsvFiles):

    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Calendar\cal_funktion3_expo.csv"
    global age_temp, age_voltage,maxXValue,exponent,uniqueZellen,df_alleZellen
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cal_funktion3_00C_expo")
    fig2 = plt.figure("cal_funktion3_10C_expo")
    fig3 = plt.figure("cal_funktion3_25C_expo")
    fig4 = plt.figure("cal_funktion3_40C_expo")
    temperatureFig = {273.15:fig1 ,283.15: fig2 ,298.15: fig3, 313.15: fig4}

    #   used to determine the max plot length to set an end to simulated data
    maxXValue = 0
    
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        exponent = 0.5

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        
        if age_temp==273.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==283.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==298.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)
        if age_temp==313.15:
            sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_voltage], ax = temperatureFig[age_temp].gca(), alpha = 0.5)

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'age_voltage':age_voltage, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()

    start = [1,1,1,0.6]
    bounds = [(-5,5),(0,1),(-1000,1000),(0.2,0.9)]
    startTime = time.time()
    print(f"Starting optimization {startTime}")
    result = optimize.minimize(funktion3_min_expo,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [0.03099997256512114, 0.13747368552595407, -999.999998685382, 0.6894569540598147]
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    
    temp = [273.15, 283.15, 298.15, 313.15]
    volt= [0.42, 2.1, 3.78, 4.2]
    errorList = []
    globalErrorList = []
    for i_t in range(0, len(temp)):
        errorList = []
        age_temp = temp[i_t]
        for i_v in range(0, len(volt)):
            age_voltage = volt[i_v]
            

            sns.lineplot(x = xdata, y = funktion3_expo(xdata, *result.x), color=colorPalette[age_temp], ax = temperatureFig[age_temp].gca(), linestyle = lineStyle[age_voltage])

            # Calculate error
            for cell in uniqueZellen:
                if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0] == age_voltage) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0] == age_temp):
                    xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                    ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                    age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
                    age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
                    
                    y_modellierte= funktion3_expo(xdata, *result.x)
                    error = (y_modellierte-ydata)**2
                    errorList.append(error*1000)
                    globalErrorList.append(error*1000)
            rsmeSOC  = np.sqrt(np.mean(errorList))
            print(f"RSME SOC age_voltage: {rsmeSOC}")
        rmseTemperature = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=0, label ="SOC")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle=':', label ="10%")
        line3 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='--', label ="50%")
        line4 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-.', label ="90%")
        line5 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-',label ="100%")
        legend1 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = "%.3f" % result.x[3]
        rmseString = "%.3f" % rmseTemperature
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = temperatureFig[age_temp].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        temperatureFig[age_temp].gca().add_artist(legend1)
        temperatureFig[age_temp].gca().add_artist(legend2)


        temperatureFig[age_temp].gca().grid(color='lightgrey', linestyle='--')
        temperatureFig[age_temp].gca().set_xlabel(f'Zeit (Tage)')
        temperatureFig[age_temp].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        temperatureFig[age_temp].gca().set_title(f'Kalendarische Modellierung - Funktion Nr. 3 bei: {age_temp-273.15}°C')




    rsme = "%.4f" % np.sqrt(np.mean(globalErrorList))
    print(f"Root Mean Squared Error: {rsme}")
    optimal_parameters = result.x
    optimal_function_value = result.fun

    csv_filename = fileToSaveValues
    header = ['Parameter {}'.format(i) for i in range(len(optimal_parameters))] + ['Optimal Function Value'] +['RSME']
    data = [np.array(optimal_parameters).tolist() + [optimal_function_value] + [np.sqrt(np.mean(globalErrorList))]]

    # Write the data to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile,delimiter =';')
        csv_writer.writerow(header)
        csv_writer.writerows(data)

    fig1.savefig(saveFigsDirectory+"cal_funktion3_00C_expo.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cal_funktion3_10C_expo.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cal_funktion3_25C_expo.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cal_funktion3_40C_expo.png", format='png',  pad_inches=0, transparent=False)


    
    plt.show()










if __name__ == '__main__':

    # fit_mutiple([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_2_S16_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_2_S18_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_3_S19_C11.csv"])



    plot_capacity_over_time(["all"])
    # fit_time_exponent(["all"])
    # NEUE DATEN EOC V2 RUN 7
    # funktion3_minimize_fixed([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_2_S18_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_2_S16_C11.csv"])
    # funktion3_minimize_expo([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_2_S18_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_2_S16_C11.csv"])
    # funktion2_minimize_fixed([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_2_S18_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_2_S16_C11.csv"])
    # funktion2_minimize_expo([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_2_S18_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_2_S16_C11.csv"])
    # funktion1_minimize_fixed([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_2_S18_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_2_S16_C11.csv"])
    # funktion1_minimize_expo([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_2_S18_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P015_2_S16_C11.csv"])

    
    # ALTE DATEN EOC V1 RUN 1
    # funktion1_minimize_fixed([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_2_S16_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_2_S18_C11.csv"])
    # funktion6_plot_and_fit_overviewGreyAndRed([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_2_S16_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_2_S18_C11.csv"],plotTemperature=283.15,plotWithFixedExponent=True,fixedExponent=0.5)
    # funktion6_plot_and_fit_overviewGreyAndRed([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_2_S16_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_2_S18_C11.csv"],plotTemperature=298.15,plotWithFixedExponent=True,fixedExponent=0.5)
    # funktion6_plot_and_fit_overviewGreyAndRed([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_2_S16_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_2_S18_C11.csv"],plotTemperature=313.15,plotWithFixedExponent=True,fixedExponent=0.5)