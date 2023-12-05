import os
import re
from itertools import groupby
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
df_alleZellen = pd.DataFrame(columns=['nameCell', 'xdata', 'ydata','age_voltage','age_temp'])
uniqueZellen = []

## HELPER FUNCTION FOR PLOTTING
filepathToDirectory = "C:\\Users\\Yannick\\bwSyncShare\\MA Yannick Fritsch\\00_Daten\\prepr_res_eoc\\"


arrayCalendarParameterID = ["P001", "P002", "P003", "P004", "P005", "P006","P007", "P008", "P009", "P010", "P011", "P012", "P013", "P014", "P015", "P016"]
arrayCyclicParameterID = ["P017", "P018", "P019", "P020", "P021", "P022", "P023", "P024", "P025", "P026", "P027", "P028", "P029", "P030", "P031", "P032", "P033", "P034", "P035", "P036", "P037", "P038", "P039", "P040", "P041", "P042", "P043", "P044", "P045", "P046", "P047", "P048", "P048", "P049", "P050", "P051", "P052", "P053", "P054", "P055", "P056", "P057", "P058", "P059", "P060", "P061", "P062", "P063", "P064"]
arrayProfileParameterID = ["P065", "P066", "P067", "P068","P069", "P070", "P071", "P072", "P073", "P074", "P075", "P076"]
ARRAY_CELLS_TO_DISPLAY = arrayCalendarParameterID


# Define the functions selected in theory part
    
def funktion1(x, a0, a1, a2):
    
    return 1 - a0 * np.exp(a1*age_voltage)*np.exp(a2/age_temp)*np.power(x, exponent)


def funktion1_expo(x, a0, a1, a2, a3):
    return 1 - a0 * np.exp(a1*age_voltage)*np.exp(a2/age_temp)*np.power(x, a3)
def funktion2(x,a0,a1):
    return 1-a0*np.exp(a1/age_voltage)*np.power(x,exponent)
def funktion2_expo(x,a0,a1,a2):
    return 1-a0*np.exp(a1/age_voltage)*np.power(x,a2)
def funktion3(x, a0, a1, a2, a3):
    return 1 - (a0 + a1*age_voltage + a2*(age_voltage*age_voltage)) * np.exp(a3/age_temp) * np.power(x, exponent)
def funktion3_expo(x, a0, a1, a2, a3,a4):
    return 1 - (a0 + a1*age_voltage + a2*(age_voltage*age_voltage)) * np.exp(a3/age_temp) * np.power(x, a4)
def funktion4(x, a0, a1, a2):
    return 1 - (a0*age_voltage+a1)*np.exp(a2/age_temp)*np.power(x,exponent)
def funktion4_expo(x, a0, a1, a2, a3):
    return 1 - (a0*age_voltage+a1)*np.exp(a2/age_temp)*np.power(x,a3)
def funktion5(x,a0,a1,a2):
    return 1- a0*np.exp(a1*age_voltage)*np.exp(a2*age_voltage/age_temp)*np.power(x,exponent)
def funktion5_expo(x,a0,a1,a2,a3):
    return 1- a0*np.exp(a1*age_voltage)*np.exp(a2*age_voltage/age_temp)*np.power(x,a3)
def funktion6(x, a0, a1,a2,a3):
    return 1-a0*np.exp(a1*age_voltage)*np.exp((a2*age_voltage+a3)/age_temp)*np.power(x,exponent)
def funktion6_expo(x, a0, a1,a2,a3,a4):
    return 1-a0*np.exp(a1*age_voltage)*np.exp((a2*age_voltage+a3)/age_temp)*np.power(x,a4)

def funktion_lookuptable(x,a0):
    return 1- a0*np.power(x,exponent)
# -------------------------------------------------------------------------------------------------------------------------------------------

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
    print(df_cell_csv_content)
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
            print(*y_values)

            for values in zip(*y_values):
                min_values.append(min(values))
                max_values.append(max(values))
                mean_values.append(sum(values) / len(values))
            # print(min_values)
            # print(max_values)
            print(f'soc: {soc}, temp: {temp}')
            print(mean_values)
            # CALCULATE RMSE
            print("mean",np.mean(y_values, axis=0))
            mean = np.mean(y_values, axis=0)
            print("standard deviation",np.std(y_values, axis=0))
            print("Mean of standard deviation",np.mean(np.std(y_values, axis=0), axis=0))
            print("rsme",np.sqrt(np.mean((mean-y_values)**2)))
                



            
            # x_values = np.array([i for i in range(0,(maxTimestamp-minTimestamp), int((maxTimestamp-minTimestamp)/(len(min_values))))])
            x_values = np.array(np.linspace(0, (maxTimestamp-minTimestamp), len(min_values), dtype=int))
            # print(x_values)
            x_values = x_values/(60*60*24)
            # print(len(min_values),len(max_values),len(mean_values),len(x_values))

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




def funktion1_plot_and_fit_overviewGreyAndRed(arrayOfPathToCsvFiles, plotTemperature=273.15, plotWithFixedExponent=False,fixedExponent=0.5):

    df_popt = pd.DataFrame(columns=['nameCell', 'a0', 'a1', 'a2', 'age_temp','age_voltage'])
    global age_temp, age_voltage, max_capacity,listOfPrograms,maxXValue,exponent
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"-.",3.78:"-", 4.2:"--"}
  
# One programm is one temp and soc level  
    listOfPrograms = []
#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0

    fig, ax = plt.subplots()
    
    xdataForMean = []
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        
        if (plotWithFixedExponent):
            exponent = fixedExponent
        else:
            exponent=funktionExponentMitWerten(age_soc)
       

        tuple_to_check = (age_temp, age_voltage, age_soc)

        if tuple_to_check not in listOfPrograms:
            print("Tuple not found in the list")
            listOfPrograms.append(tuple_to_check)

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        print(age_voltage, age_temp)
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        print(xdata)
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        print(ydata)

        # p0 = [0.2, 0.1, -0.027, -20]
        # lower_bounds = [-100, -100, -1, -22]
        # upper_bounds = [100, 100, 0, -18]

        # popt, pcov = optimize.curve_fit(func, xdata, ydata, p0=p0, bounds=[lower_bounds, upper_bounds])


        p0 = [1,1,1]
        lower_bounds = [-5, -5, -1000]
        upper_bounds = [5, 5, 1000]
        popt, pcov = optimize.curve_fit(funktion1, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
        print("popt", popt)
        # print("xdata", xdata, "ydata", ydata, "*popt",
        #       popt, "func", func(xdata, *popt))
        if (age_temp == plotTemperature):
            ax.plot(xdata, ydata, color='lightgray',linestyle=lineStyle[age_voltage])
        # ax.plot(xdata, funktion1(xdata, *popt), 'r-', label='fitting')
        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"

        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'a0': popt[0], 'a1': popt[1], 'a2': popt[2], 'age_temp':age_temp, 'age_voltage': age_voltage, 'age_soc': age_soc,}
        else:
            new_row = {'nameCell': "unknown",
                       'a0': popt[0], 'a1': popt[1], 'a2': popt[2], 'age_temp':"ERROR", 'age_voltage': "ERROR"}
        new_df = pd.DataFrame(new_row, index=[0])
        df_popt = pd.concat([df_popt, new_df], ignore_index=True)
        # subfigure.set_xlabel('timestamp_s')
        # subfigure.set_ylabel('cap_aged_est_Ah')
    global_meanValues = df_popt.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean().values
    print("df_popt",df_popt.sort_values("age_soc"))
    print('global mean values', global_meanValues)
    x_data = np.linspace(0, maxXValue, 20)
    exponent = 0.5
    age_temp = 273.15
    age_voltage = 0.42
    ax.plot(x_data, funktion1(x_data, *global_meanValues), color = 'red',linestyle=lineStyle[age_voltage])
    age_voltage = 2.1
    ax.plot(x_data, funktion1(x_data, *global_meanValues), color = 'red',linestyle=lineStyle[age_voltage])
    age_voltage = 3.78
    ax.plot(x_data, funktion1(x_data, *global_meanValues), color = 'red',linestyle=lineStyle[age_voltage])
    age_voltage = 4.2
    ax.plot(x_data, funktion1(x_data, *global_meanValues), color = 'red',linestyle=lineStyle[age_voltage])

    plt.show()
    averageArray=[]
    for tuple in listOfPrograms:
        
        print("tuple",tuple)
        if (tuple[0] == plotTemperature):
            
            age_temperature = plotTemperature #tuple[0]
            age_voltage = tuple[1]
            selected_rows = df_popt[(df_popt['age_temp'] == age_temperature) & (df_popt['age_voltage'] == age_voltage)]

            #Calculate average values of all rows
            average_values = selected_rows.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean()
            print("local mean values", average_values.values)
            
            if (plotWithFixedExponent):
                exponent = fixedExponent
            else:
                exponent=funktionExponentMitWerten(age_soc)
            print("Exponent", exponent)
            # x_data = x_data/(60*60*24)
            x_data = np.linspace(0, maxXValue, 40)
            y_data = funktion1(x_data, *average_values)
            ax.plot(x_data, y_data , color=colorPalette[age_temperature],linestyle=lineStyle[age_voltage])
            
            

            # ERROR BERECHNEN 
            x_data = np.linspace(0, maxXValue, 20)
            temperature = int(tuple[0]-273.15)
            soc = tuple[2]
            filename = f"avg_SOC_{soc}_T_{temperature}.csv"

            xvalues = []
            mean_values = []

            with open(tempFilesDirectory+filename, mode='r') as file:
                reader = csv.reader(file, delimiter=';')
                header = next(reader)  # Skip the header row
                for row in reader:
                    xvalues.append(float(row[0]))
                    mean_values.append(float(row[1]))
            #  Nochmal x und y berechnen, damit die selbe Anzahl wie average in funktion plot_capacity_over_time gibt
            maxXValue = max(xvalues)
            x_data = np.linspace(0, maxXValue, len(mean_values))
            y_data = y_data = funktion1(x_data, *average_values)
            # CALCULATE ERROR
            # print("standard deviation",np.std(y_data, axis=0))
            # print("Mean of standard deviation",np.mean(np.std(y_data)))
            # print("RSME",np.sqrt(np.mean((mean_values-y_data)**2)))
            averageArray.append(np.sqrt(np.mean((mean_values-y_data)**2)))
    print(f"Total RSME: {np.mean(averageArray)*100}%")
    rsme2 = "%.2f" % (np.mean(averageArray)*100)
    # custom_lines = [Line2D([0], [0], color="grey", lw=2),
    #             Line2D([0], [0], color=colorPalette[age_temperature], lw=2), Line2D([0], [0], color=colorPalette[age_temperature], lw=0)]
    # plt.legend(custom_lines, ['Messungen', 'Modellierung', f" RSME: {rsme2}%"])
    line1 = plt.Line2D([0], [0], color="grey", lw=0, label ="SOC")
    line2 = plt.Line2D([0], [0], color="grey", lw=2,linestyle=':', label ="10%")
    line3 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-.', label ="50%")
    line4 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-', label ="90%")
    line5 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='--',label ="100%")
    legend1 = plt.legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

    line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
    line2 = plt.Line2D([0], [0], color=colorPalette[age_temperature], lw=2, label = "Modellierung")
    line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RSME: {rsme2}%")
    legend2 = plt.legend(handles=[line1, line2, line3], loc='upper right')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)


    plt.grid(color='lightgrey', linestyle='--')
    plt.xlabel(f'Zeit (Tage)')
    plt.ylabel('Relative Kapazität C(t)/C_init (-)')
    ax.set_title(f'Kalendarische Modellierung - Funktion Nr. 1 bei: {plotTemperature-273.15}°C')
    # plt.text(-0.1, -0.2, f'', transform=plt.gca().transAxes)

    plt.show()

def funktion2_plot_and_fit_overviewGreyAndRed(arrayOfPathToCsvFiles, plotTemperature=273.15, plotWithFixedExponent=False,fixedExponent=0.5):
    df_popt = pd.DataFrame(columns=['nameCell', 'a0', 'a1','age_temp','age_voltage'])
    global age_temp, age_voltage, max_capacity,listOfPrograms,maxXValue,exponent
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"-.",3.78:"-", 4.2:"--"}
  
# One programm is one temp and soc level  
    listOfPrograms = []
#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0

    fig, ax = plt.subplots()
    
    xdataForMean = []
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        if (plotWithFixedExponent):
            exponent = fixedExponent
        else:
            exponent=funktionExponentMitWerten(age_soc)
        tuple_to_check = (age_temp, age_voltage, age_soc)

        if tuple_to_check not in listOfPrograms:
            print("Tuple not found in the list")
            listOfPrograms.append(tuple_to_check)

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        print(age_voltage, age_temp)
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        print(xdata)
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        print(ydata)

        # p0 = [0.2, 0.1, -0.027, -20]
        # lower_bounds = [-100, -100, -1, -22]
        # upper_bounds = [100, 100, 0, -18]

        # popt, pcov = optimize.curve_fit(func, xdata, ydata, p0=p0, bounds=[lower_bounds, upper_bounds])


        p0 = [1,1]
        lower_bounds = [-5, -5]
        upper_bounds = [5, 5]
        popt, pcov = optimize.curve_fit(funktion2, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
        print("popt", popt)
        # print("xdata", xdata, "ydata", ydata, "*popt",
        #       popt, "func", func(xdata, *popt))
        if (age_temp == plotTemperature):
            ax.plot(xdata, ydata, color='lightgray',linestyle=lineStyle[age_voltage])
        # ax.plot(xdata, funktion1(xdata, *popt), 'r-', label='fitting')
        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"

        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'a0': popt[0], 'a1': popt[1], 'age_temp':age_temp, 'age_voltage': age_voltage, 'age_soc': age_soc,}
        else:
            new_row = {'nameCell': "unknown",
                       'a0': popt[0], 'a1': popt[1], 'age_temp':"ERROR", 'age_voltage': "ERROR"}
        new_df = pd.DataFrame(new_row, index=[0])
        df_popt = pd.concat([df_popt, new_df], ignore_index=True)
        # subfigure.set_xlabel('timestamp_s')
        # subfigure.set_ylabel('cap_aged_est_Ah')
    global_meanValues = df_popt.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean().values
    print("df_popt",df_popt.sort_values("age_soc"))
    print('global mean values', global_meanValues)
    x_data = np.linspace(0, maxXValue, 20)
    # ax.plot(x_data, funktion1(x_data, *global_meanValues), color = 'red',linestyle=lineStyle[age_voltage])
    averageArray=[]
    for tuple in listOfPrograms:
        
        print("tuple",tuple)
        if (tuple[0] == plotTemperature):
            
            age_temperature = plotTemperature #tuple[0]
            age_voltage = tuple[1]
            selected_rows = df_popt[(df_popt['age_temp'] == age_temperature) & (df_popt['age_voltage'] == age_voltage)]

            #Calculate average values of all rows
            average_values = selected_rows.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean()
            print("local mean values", average_values.values)
            
            if (plotWithFixedExponent):
                exponent = fixedExponent
            else:
                exponent=funktionExponentMitWerten(age_soc)

            print("Exponent", exponent)
            # x_data = x_data/(60*60*24)
            x_data = np.linspace(0, maxXValue, 40)
            y_data = funktion2(x_data, *average_values)
            ax.plot(x_data, y_data , color=colorPalette[age_temperature],linestyle=lineStyle[age_voltage])
            

            # ERROR BERECHNEN 
            x_data = np.linspace(0, maxXValue, 20)
            temperature = int(tuple[0]-273.15)
            soc = tuple[2]
            filename = f"avg_SOC_{soc}_T_{temperature}.csv"

            xvalues = []
            mean_values = []

            with open(tempFilesDirectory+filename, mode='r') as file:
                reader = csv.reader(file, delimiter=';')
                header = next(reader)  # Skip the header row
                for row in reader:
                    xvalues.append(float(row[0]))
                    mean_values.append(float(row[1]))
            #  Nochmal x und y berechnen, damit die selbe Anzahl wie average in funktion plot_capacity_over_time gibt
            maxXValue = max(xvalues)
            x_data = np.linspace(0, maxXValue, len(mean_values))
            y_data = y_data = funktion2(x_data, *average_values)
            # CALCULATE ERROR
            # print("standard deviation",np.std(y_data, axis=0))
            # print("Mean of standard deviation",np.mean(np.std(y_data)))
            # print("RSME",np.sqrt(np.mean((mean_values-y_data)**2)))
            averageArray.append(np.sqrt(np.mean((mean_values-y_data)**2)))
    print(f"Total RSME: {np.mean(averageArray)*100}%")
    rsme2 = "%.2f" % (np.mean(averageArray)*100)
    # custom_lines = [Line2D([0], [0], color="grey", lw=2),
    #             Line2D([0], [0], color=colorPalette[age_temperature], lw=2), Line2D([0], [0], color=colorPalette[age_temperature], lw=0)]
    # plt.legend(custom_lines, ['Messungen', 'Modellierung', f" RSME: {rsme2}%"])
    line1 = plt.Line2D([0], [0], color="grey", lw=0, label ="SOC")
    line2 = plt.Line2D([0], [0], color="grey", lw=2,linestyle=':', label ="10%")
    line3 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-.', label ="50%")
    line4 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-', label ="90%")
    line5 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='--',label ="100%")
    legend1 = plt.legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

    line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
    line2 = plt.Line2D([0], [0], color=colorPalette[age_temperature], lw=2, label = "Modellierung")
    line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RSME: {rsme2}%")
    legend2 = plt.legend(handles=[line1, line2, line3], loc='upper right')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)


    plt.grid(color='lightgrey', linestyle='--')
    plt.xlabel(f'Zeit (Tage)')
    plt.ylabel('Relative Kapazität C(t)/C_init (-)')
    ax.set_title(f'Kalendarische Modellierung - Funktion Nr. 2 bei: {plotTemperature}°K')

    plt.show()

def funktion3_plot_and_fit_overviewGreyAndRed(arrayOfPathToCsvFiles, plotTemperature=273.15, plotWithFixedExponent=False,fixedExponent=0.5):
    df_popt = pd.DataFrame(columns=['nameCell', 'a0', 'a1','a2','a3','age_temp','age_voltage'])
    global age_temp, age_voltage, max_capacity,listOfPrograms,maxXValue,exponent
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"-.",3.78:"-", 4.2:"--"}
  
# One programm is one temp and soc level  
    listOfPrograms = []
#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0

    fig, ax = plt.subplots()
    
    xdataForMean = []
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        if (plotWithFixedExponent):
            exponent = fixedExponent
        else:
            exponent=funktionExponentMitWerten(age_soc)
        tuple_to_check = (age_temp, age_voltage, age_soc)

        if tuple_to_check not in listOfPrograms:
            print("Tuple not found in the list")
            listOfPrograms.append(tuple_to_check)

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        print(age_voltage, age_temp)
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        print(xdata)
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        print(ydata)

        # p0 = [0.2, 0.1, -0.027, -20]
        # lower_bounds = [-100, -100, -1, -22]
        # upper_bounds = [100, 100, 0, -18]

        # popt, pcov = optimize.curve_fit(func, xdata, ydata, p0=p0, bounds=[lower_bounds, upper_bounds])


        p0 = [1,1,1,1]
        lower_bounds = [-5, -5, -1000, -100]
        upper_bounds = [5, 5, 100,100]
        popt, pcov = optimize.curve_fit(funktion3, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
        print("popt", popt)
        # print("xdata", xdata, "ydata", ydata, "*popt",
        #       popt, "func", func(xdata, *popt))
        if (age_temp == plotTemperature):
            ax.plot(xdata, ydata, color='lightgray',linestyle=lineStyle[age_voltage])
        # ax.plot(xdata, funktion1(xdata, *popt), 'r-', label='fitting')
        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"

        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'a0': popt[0], 'a1': popt[1],'a2': popt[2],'a3': popt[3], 'age_temp':age_temp, 'age_voltage': age_voltage, 'age_soc': age_soc,}
        else:
            new_row = {'nameCell': "unknown",
                       'a0': popt[0], 'a1': popt[1],'a2': popt[2],'a3': popt[3], 'age_temp':"ERROR", 'age_voltage': "ERROR"}
        new_df = pd.DataFrame(new_row, index=[0])
        df_popt = pd.concat([df_popt, new_df], ignore_index=True)
        # subfigure.set_xlabel('timestamp_s')
        # subfigure.set_ylabel('cap_aged_est_Ah')
    global_meanValues = df_popt.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean().values
    print("df_popt",df_popt.sort_values("age_soc"))
    print('global mean values', global_meanValues)
    x_data = np.linspace(0, maxXValue, 20)
    # ax.plot(x_data, funktion1(x_data, *global_meanValues), color = 'red',linestyle=lineStyle[age_voltage])
    averageArray=[]
    for tuple in listOfPrograms:
        
        print("tuple",tuple)
        if (tuple[0] == plotTemperature):
            
            age_temperature = plotTemperature #tuple[0]
            age_voltage = tuple[1]
            selected_rows = df_popt[(df_popt['age_temp'] == age_temperature) & (df_popt['age_voltage'] == age_voltage)]

            #Calculate average values of all rows
            average_values = selected_rows.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean()
            print("local mean values", average_values.values)
            
            if (plotWithFixedExponent):
                exponent = fixedExponent
            else:
                exponent=funktionExponentMitWerten(age_soc)
            print("Exponent", exponent)
            # x_data = x_data/(60*60*24)
            x_data = np.linspace(0, maxXValue, 40)
            y_data = funktion3(x_data, *average_values)
            ax.plot(x_data, y_data , color=colorPalette[age_temperature],linestyle=lineStyle[age_voltage])
            

            # ERROR BERECHNEN 
            x_data = np.linspace(0, maxXValue, 20)
            temperature = int(tuple[0]-273.15)
            soc = tuple[2]
            filename = f"avg_SOC_{soc}_T_{temperature}.csv"

            xvalues = []
            mean_values = []

            with open(tempFilesDirectory+filename, mode='r') as file:
                reader = csv.reader(file, delimiter=';')
                header = next(reader)  # Skip the header row
                for row in reader:
                    xvalues.append(float(row[0]))
                    mean_values.append(float(row[1]))
            #  Nochmal x und y berechnen, damit die selbe Anzahl wie average in funktion plot_capacity_over_time gibt
            maxXValue = max(xvalues)
            x_data = np.linspace(0, maxXValue, len(mean_values))
            y_data = y_data = funktion3(x_data, *average_values)
            # CALCULATE ERROR
            # print("standard deviation",np.std(y_data, axis=0))
            # print("Mean of standard deviation",np.mean(np.std(y_data)))
            # print("RSME",np.sqrt(np.mean((mean_values-y_data)**2)))
            averageArray.append(np.sqrt(np.mean((mean_values-y_data)**2)))
    print(f"Total RSME: {np.mean(averageArray)*100}%")
    rsme2 = "%.2f" % (np.mean(averageArray)*100)
    # custom_lines = [Line2D([0], [0], color="grey", lw=2),
    #             Line2D([0], [0], color=colorPalette[age_temperature], lw=2), Line2D([0], [0], color=colorPalette[age_temperature], lw=0)]
    # plt.legend(custom_lines, ['Messungen', 'Modellierung', f" RSME: {rsme2}%"])
    line1 = plt.Line2D([0], [0], color="grey", lw=0, label ="SOC")
    line2 = plt.Line2D([0], [0], color="grey", lw=2,linestyle=':', label ="10%")
    line3 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-.', label ="50%")
    line4 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-', label ="90%")
    line5 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='--',label ="100%")
    legend1 = plt.legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

    line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
    line2 = plt.Line2D([0], [0], color=colorPalette[age_temperature], lw=2, label = "Modellierung")
    line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RSME: {rsme2}%")
    legend2 = plt.legend(handles=[line1, line2, line3], loc='upper right')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)


    plt.grid(color='lightgrey', linestyle='--')
    plt.xlabel(f'Zeit (Tage)')
    plt.ylabel('Relative Kapazität C(t)/C_init (-)')
    ax.set_title(f'Kalendarische Modellierung - Funktion Nr. 3 bei: {plotTemperature}°K')

    plt.show()

def funktion4_plot_and_fit_overviewGreyAndRed(arrayOfPathToCsvFiles, plotTemperature=273.15, plotWithFixedExponent=False,fixedExponent=0.5):
    df_popt = pd.DataFrame(columns=['nameCell', 'a0', 'a1','a2','age_temp','age_voltage'])
    global age_temp, age_voltage, max_capacity,listOfPrograms,maxXValue,exponent
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"-.",3.78:"-", 4.2:"--"}
  
# One programm is one temp and soc level  
    listOfPrograms = []
#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0

    fig, ax = plt.subplots()
    
    xdataForMean = []
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        if (plotWithFixedExponent):
            exponent = fixedExponent
        else:
            exponent=funktionExponentMitWerten(age_soc)
        tuple_to_check = (age_temp, age_voltage, age_soc)

        if tuple_to_check not in listOfPrograms:
            print("Tuple not found in the list")
            listOfPrograms.append(tuple_to_check)

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        print(age_voltage, age_temp)
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        print(xdata)
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        print(ydata)

        # p0 = [0.2, 0.1, -0.027, -20]
        # lower_bounds = [-100, -100, -1, -22]
        # upper_bounds = [100, 100, 0, -18]

        # popt, pcov = optimize.curve_fit(func, xdata, ydata, p0=p0, bounds=[lower_bounds, upper_bounds])


        p0 = [1,1,1]
        lower_bounds = [-5, -5, -1000]
        upper_bounds = [5, 5, 100]
        popt, pcov = optimize.curve_fit(funktion4, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
        print("popt", popt)
        # print("xdata", xdata, "ydata", ydata, "*popt",
        #       popt, "func", func(xdata, *popt))
        if (age_temp == plotTemperature):
            ax.plot(xdata, ydata, color='lightgray',linestyle=lineStyle[age_voltage])
        # ax.plot(xdata, funktion1(xdata, *popt), 'r-', label='fitting')
        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"

        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'a0': popt[0], 'a1': popt[1],'a2': popt[2], 'age_temp':age_temp, 'age_voltage': age_voltage, 'age_soc': age_soc,}
        else:
            new_row = {'nameCell': "unknown",
                       'a0': popt[0], 'a1': popt[1],'a2': popt[2], 'age_temp':"ERROR", 'age_voltage': "ERROR"}
        new_df = pd.DataFrame(new_row, index=[0])
        df_popt = pd.concat([df_popt, new_df], ignore_index=True)
        # subfigure.set_xlabel('timestamp_s')
        # subfigure.set_ylabel('cap_aged_est_Ah')
    global_meanValues = df_popt.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean().values
    print("df_popt",df_popt.sort_values("age_soc"))
    print('global mean values', global_meanValues)
    x_data = np.linspace(0, maxXValue, 20)
    # ax.plot(x_data, funktion1(x_data, *global_meanValues), color = 'red',linestyle=lineStyle[age_voltage])
    averageArray=[]
    for tuple in listOfPrograms:
        
        print("tuple",tuple)
        if (tuple[0] == plotTemperature):
            
            age_temperature = plotTemperature #tuple[0]
            age_voltage = tuple[1]
            selected_rows = df_popt[(df_popt['age_temp'] == age_temperature) & (df_popt['age_voltage'] == age_voltage)]

            #Calculate average values of all rows
            average_values = selected_rows.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean()
            print("local mean values", average_values.values)
            
            if (plotWithFixedExponent):
                exponent = fixedExponent
            else:
                exponent=funktionExponentMitWerten(age_soc)
            print("Exponent", exponent)
            # x_data = x_data/(60*60*24)
            x_data = np.linspace(0, maxXValue, 40)
            y_data = funktion4(x_data, *average_values)
            ax.plot(x_data, y_data , color=colorPalette[age_temperature],linestyle=lineStyle[age_voltage])
            

            # ERROR BERECHNEN 
            x_data = np.linspace(0, maxXValue, 20)
            temperature = int(tuple[0]-273.15)
            soc = tuple[2]
            filename = f"avg_SOC_{soc}_T_{temperature}.csv"

            xvalues = []
            mean_values = []

            with open(tempFilesDirectory+filename, mode='r') as file:
                reader = csv.reader(file, delimiter=';')
                header = next(reader)  # Skip the header row
                for row in reader:
                    xvalues.append(float(row[0]))
                    mean_values.append(float(row[1]))
            #  Nochmal x und y berechnen, damit die selbe Anzahl wie average in funktion plot_capacity_over_time gibt
            maxXValue = max(xvalues)
            x_data = np.linspace(0, maxXValue, len(mean_values))
            y_data = y_data = funktion4(x_data, *average_values)
            # CALCULATE ERROR
            # print("standard deviation",np.std(y_data, axis=0))
            # print("Mean of standard deviation",np.mean(np.std(y_data)))
            # print("RSME",np.sqrt(np.mean((mean_values-y_data)**2)))
            averageArray.append(np.sqrt(np.mean((mean_values-y_data)**2)))
    print(f"Total RSME: {np.mean(averageArray)*100}%")
    rsme2 = "%.2f" % (np.mean(averageArray)*100)
    # custom_lines = [Line2D([0], [0], color="grey", lw=2),
    #             Line2D([0], [0], color=colorPalette[age_temperature], lw=2), Line2D([0], [0], color=colorPalette[age_temperature], lw=0)]
    # plt.legend(custom_lines, ['Messungen', 'Modellierung', f" RSME: {rsme2}%"])
    line1 = plt.Line2D([0], [0], color="grey", lw=0, label ="SOC")
    line2 = plt.Line2D([0], [0], color="grey", lw=2,linestyle=':', label ="10%")
    line3 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-.', label ="50%")
    line4 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-', label ="90%")
    line5 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='--',label ="100%")
    legend1 = plt.legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

    line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
    line2 = plt.Line2D([0], [0], color=colorPalette[age_temperature], lw=2, label = "Modellierung")
    line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RSME: {rsme2}%")
    legend2 = plt.legend(handles=[line1, line2, line3], loc='upper right')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)


    plt.grid(color='lightgrey', linestyle='--')
    plt.xlabel(f'Zeit (Tage)')
    plt.ylabel('Relative Kapazität C(t)/C_init (-)')
    ax.set_title(f'Kalendarische Modellierung - Funktion Nr. 4 bei: {plotTemperature}°K')

    plt.show()

def funktion5_plot_and_fit_overviewGreyAndRed(arrayOfPathToCsvFiles, plotTemperature=273.15, plotWithFixedExponent=False,fixedExponent=0.5):
    df_popt = pd.DataFrame(columns=['nameCell', 'a0', 'a1','a2','age_temp','age_voltage'])
    global age_temp, age_voltage, max_capacity,listOfPrograms,maxXValue,exponent
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"-.",3.78:"-", 4.2:"--"}
  
# One programm is one temp and soc level  
    listOfPrograms = []
#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0

    fig, ax = plt.subplots()
    
    xdataForMean = []
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        if (plotWithFixedExponent):
            exponent = fixedExponent
        else:
            exponent=funktionExponentMitWerten(age_soc)

        tuple_to_check = (age_temp, age_voltage, age_soc)

        if tuple_to_check not in listOfPrograms:
            print("Tuple not found in the list")
            listOfPrograms.append(tuple_to_check)

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        print(age_voltage, age_temp)
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        print(xdata)
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        print(ydata)

        # p0 = [0.2, 0.1, -0.027, -20]
        # lower_bounds = [-100, -100, -1, -22]
        # upper_bounds = [100, 100, 0, -18]

        # popt, pcov = optimize.curve_fit(func, xdata, ydata, p0=p0, bounds=[lower_bounds, upper_bounds])


        p0 = [1,1,1]
        lower_bounds = [-5, -5, -1000]
        upper_bounds = [5, 5, 100]
        popt, pcov = optimize.curve_fit(funktion5, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
        print("popt", popt)
        # print("xdata", xdata, "ydata", ydata, "*popt",
        #       popt, "func", func(xdata, *popt))
        if (age_temp == plotTemperature):
            ax.plot(xdata, ydata, color='lightgray',linestyle=lineStyle[age_voltage])
        # ax.plot(xdata, funktion1(xdata, *popt), 'r-', label='fitting')
        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"

        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'a0': popt[0], 'a1': popt[1],'a2': popt[2], 'age_temp':age_temp, 'age_voltage': age_voltage, 'age_soc': age_soc,}
        else:
            new_row = {'nameCell': "unknown",
                       'a0': popt[0], 'a1': popt[1],'a2': popt[2], 'age_temp':"ERROR", 'age_voltage': "ERROR"}
        new_df = pd.DataFrame(new_row, index=[0])
        df_popt = pd.concat([df_popt, new_df], ignore_index=True)
        # subfigure.set_xlabel('timestamp_s')
        # subfigure.set_ylabel('cap_aged_est_Ah')
    global_meanValues = df_popt.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean().values
    print("df_popt",df_popt.sort_values("age_soc"))
    print('global mean values', global_meanValues)
    x_data = np.linspace(0, maxXValue, 20)
    # ax.plot(x_data, funktion1(x_data, *global_meanValues), color = 'red',linestyle=lineStyle[age_voltage])
    averageArray=[]
    for tuple in listOfPrograms:
        
        print("tuple",tuple)
        if (tuple[0] == plotTemperature):
            
            age_temperature = plotTemperature #tuple[0]
            age_voltage = tuple[1]
            selected_rows = df_popt[(df_popt['age_temp'] == age_temperature) & (df_popt['age_voltage'] == age_voltage)]

            #Calculate average values of all rows
            average_values = selected_rows.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean()
            print("local mean values", average_values.values)
            
            if (plotWithFixedExponent):
                exponent = fixedExponent
            else:
                exponent=funktionExponentMitWerten(age_soc)
            print("Exponent", exponent)
            # x_data = x_data/(60*60*24)
            x_data = np.linspace(0, maxXValue, 40)
            y_data = funktion5(x_data, *average_values)
            ax.plot(x_data, y_data , color=colorPalette[age_temperature],linestyle=lineStyle[age_voltage])
            

            # ERROR BERECHNEN 
            x_data = np.linspace(0, maxXValue, 20)
            temperature = int(tuple[0]-273.15)
            soc = tuple[2]
            filename = f"avg_SOC_{soc}_T_{temperature}.csv"

            xvalues = []
            mean_values = []

            with open(tempFilesDirectory+filename, mode='r') as file:
                reader = csv.reader(file, delimiter=';')
                header = next(reader)  # Skip the header row
                for row in reader:
                    xvalues.append(float(row[0]))
                    mean_values.append(float(row[1]))
            #  Nochmal x und y berechnen, damit die selbe Anzahl wie average in funktion plot_capacity_over_time gibt
            maxXValue = max(xvalues)
            x_data = np.linspace(0, maxXValue, len(mean_values))
            y_data = y_data = funktion5(x_data, *average_values)
            # CALCULATE ERROR
            # print("standard deviation",np.std(y_data, axis=0))
            # print("Mean of standard deviation",np.mean(np.std(y_data)))
            # print("RSME",np.sqrt(np.mean((mean_values-y_data)**2)))
            averageArray.append(np.sqrt(np.mean((mean_values-y_data)**2)))
    print(f"Total RSME: {np.mean(averageArray)*100}%")
    rsme2 = "%.2f" % (np.mean(averageArray)*100)
    # custom_lines = [Line2D([0], [0], color="grey", lw=2),
    #             Line2D([0], [0], color=colorPalette[age_temperature], lw=2), Line2D([0], [0], color=colorPalette[age_temperature], lw=0)]
    # plt.legend(custom_lines, ['Messungen', 'Modellierung', f" RSME: {rsme2}%"])
    line1 = plt.Line2D([0], [0], color="grey", lw=0, label ="SOC")
    line2 = plt.Line2D([0], [0], color="grey", lw=2,linestyle=':', label ="10%")
    line3 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-.', label ="50%")
    line4 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-', label ="90%")
    line5 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='--',label ="100%")
    legend1 = plt.legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

    line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
    line2 = plt.Line2D([0], [0], color=colorPalette[age_temperature], lw=2, label = "Modellierung")
    line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RSME: {rsme2}%")
    legend2 = plt.legend(handles=[line1, line2, line3], loc='upper right')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)


    plt.grid(color='lightgrey', linestyle='--')
    plt.xlabel(f'Zeit (Tage)')
    plt.ylabel('Relative Kapazität C(t)/C_init (-)')
    ax.set_title(f'Kalendarische Modellierung - Funktion Nr. 5 bei: {plotTemperature}°K')

    plt.show()

def funktion6_plot_and_fit_overviewGreyAndRed(arrayOfPathToCsvFiles, plotTemperature=273.15, plotWithFixedExponent=False,fixedExponent=0.5):
    df_popt = pd.DataFrame(columns=['nameCell', 'a0', 'a1','a2','a3','age_temp','age_voltage'])
    global age_temp, age_voltage, max_capacity,listOfPrograms,maxXValue,exponent
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"-.",3.78:"-", 4.2:"--"}
  
# One programm is one temp and soc level  
    listOfPrograms = []
#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0

    fig, ax = plt.subplots()
    
    xdataForMean = []
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        if (plotWithFixedExponent):
            exponent = fixedExponent
        else:
            exponent=funktionExponentMitWerten(age_soc)

        tuple_to_check = (age_temp, age_voltage, age_soc)

        if tuple_to_check not in listOfPrograms:
            print("Tuple not found in the list")
            listOfPrograms.append(tuple_to_check)

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        print(age_voltage, age_temp)
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        print(xdata)
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        print(ydata)

        # p0 = [0.2, 0.1, -0.027, -20]
        # lower_bounds = [-100, -100, -1, -22]
        # upper_bounds = [100, 100, 0, -18]

        # popt, pcov = optimize.curve_fit(func, xdata, ydata, p0=p0, bounds=[lower_bounds, upper_bounds])


        p0 = [1,1,1,1]
        lower_bounds = [-5, -5, -1000,-100]
        upper_bounds = [5, 5, 100,100]
        popt, pcov = optimize.curve_fit(funktion6, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
        print("popt", popt)
        # print("xdata", xdata, "ydata", ydata, "*popt",
        #       popt, "func", func(xdata, *popt))
        if (age_temp == plotTemperature):
            ax.plot(xdata, ydata, color='lightgray',linestyle=lineStyle[age_voltage])
        # ax.plot(xdata, funktion1(xdata, *popt), 'r-', label='fitting')
        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"

        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'a0': popt[0], 'a1': popt[1],'a2': popt[2],'a3': popt[3], 'age_temp':age_temp, 'age_voltage': age_voltage, 'age_soc': age_soc,}
        else:
            new_row = {'nameCell': "unknown",
                       'a0': popt[0], 'a1': popt[1],'a2': popt[2],'a3': popt[3], 'age_temp':"ERROR", 'age_voltage': "ERROR"}
        new_df = pd.DataFrame(new_row, index=[0])
        df_popt = pd.concat([df_popt, new_df], ignore_index=True)
        # subfigure.set_xlabel('timestamp_s')
        # subfigure.set_ylabel('cap_aged_est_Ah')
    global_meanValues = df_popt.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean().values
    print("df_popt",df_popt.sort_values("age_soc"))
    print('global mean values', global_meanValues)
    x_data = np.linspace(0, maxXValue, 20)
    # ax.plot(x_data, funktion1(x_data, *global_meanValues), color = 'red',linestyle=lineStyle[age_voltage])
    averageArray=[]
    for tuple in listOfPrograms:
        
        print("tuple",tuple)
        if (tuple[0] == plotTemperature):
            
            age_temperature = plotTemperature #tuple[0]
            age_voltage = tuple[1]
            selected_rows = df_popt[(df_popt['age_temp'] == age_temperature) & (df_popt['age_voltage'] == age_voltage)]

            #Calculate average values of all rows
            average_values = selected_rows.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean()
            print("local mean values", average_values.values)
            
            if (plotWithFixedExponent):
                exponent = fixedExponent
            else:
                exponent=funktionExponentMitWerten(age_soc)
            print("Exponent", exponent)
            # x_data = x_data/(60*60*24)
            x_data = np.linspace(0, maxXValue, 40)
            y_data = funktion6(x_data, *average_values)
            ax.plot(x_data, y_data , color=colorPalette[age_temperature],linestyle=lineStyle[age_voltage])
            

            # ERROR BERECHNEN 
            x_data = np.linspace(0, maxXValue, 20)
            temperature = int(tuple[0]-273.15)
            soc = tuple[2]
            filename = f"avg_SOC_{soc}_T_{temperature}.csv"

            xvalues = []
            mean_values = []

            with open(tempFilesDirectory+filename, mode='r') as file:
                reader = csv.reader(file, delimiter=';')
                header = next(reader)  # Skip the header row
                for row in reader:
                    xvalues.append(float(row[0]))
                    mean_values.append(float(row[1]))
            #  Nochmal x und y berechnen, damit die selbe Anzahl wie average in funktion plot_capacity_over_time gibt
            maxXValue = max(xvalues)
            x_data = np.linspace(0, maxXValue, len(mean_values))
            y_data = y_data = funktion6(x_data, *average_values)
            # CALCULATE ERROR
            # print("standard deviation",np.std(y_data, axis=0))
            # print("Mean of standard deviation",np.mean(np.std(y_data)))
            # print("RSME",np.sqrt(np.mean((mean_values-y_data)**2)))
            averageArray.append(np.sqrt(np.mean((mean_values-y_data)**2)))
    print(f"Total RSME: {np.mean(averageArray)*100}%")
    rsme2 = "%.2f" % (np.mean(averageArray)*100)
    # custom_lines = [Line2D([0], [0], color="grey", lw=2),
    #             Line2D([0], [0], color=colorPalette[age_temperature], lw=2), Line2D([0], [0], color=colorPalette[age_temperature], lw=0)]
    # plt.legend(custom_lines, ['Messungen', 'Modellierung', f" RSME: {rsme2}%"])
    line1 = plt.Line2D([0], [0], color="grey", lw=0, label ="SOC")
    line2 = plt.Line2D([0], [0], color="grey", lw=2,linestyle=':', label ="10%")
    line3 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-.', label ="50%")
    line4 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-', label ="90%")
    line5 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='--',label ="100%")
    legend1 = plt.legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

    line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
    line2 = plt.Line2D([0], [0], color=colorPalette[age_temperature], lw=2, label = "Modellierung")
    line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RSME: {rsme2}%")
    legend2 = plt.legend(handles=[line1, line2, line3], loc='upper right')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)


    plt.grid(color='lightgrey', linestyle='--')
    plt.xlabel(f'Zeit (Tage)')
    plt.ylabel('Relative Kapazität C(t)/C_init (-)')
    ax.set_title(f'Kalendarische Modellierung - Funktion Nr. 6 bei: {plotTemperature}°K')

    plt.show()

def funktion1_min(parameters):
    global uniqueZellen,df_alleZellen,age_voltage,age_temp
    # Für jede Zelle
    a0= parameters[0]
    a1= parameters[1]
    a2= parameters[2]
    a3= parameters[3] 
    summeErrors=0
    print(".")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
        age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
        
        y_modellierte= funktion1_expo(xdata, a0, a1, a2, a3)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors

def funktion1_minimize(arrayOfPathToCsvFiles):

    
    global age_temp, age_voltage,maxXValue,exponent,uniqueZellen,df_alleZellen
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {0.42:":",2.1:"-.",3.78:"-", 4.2:"--"}
  

#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0
    fig, axes = plt.subplots(1, 4)
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        exponent = 0.5

        
        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        print(age_voltage, age_temp)
        # Get relative time in seconds and not absolute timestamps
        relevant_df.loc[:, 'timestamp_s'] = relevant_df['timestamp_s'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        # max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        # Load the data from the pandas dataframe
        xdata = relevant_df['timestamp_s'].values
        # Rechne die X Achse in Tage um statt Sekunden
        xdata = xdata/(60*60*24)
        print(xdata)
        if max(xdata) > maxXValue:
            maxXValue = max(xdata)

        ydata = relevant_df['cap_aged_est_Ah'].values
        print(ydata)
        if age_temp==273.15:
            sns.lineplot(x = xdata, y = ydata, color = colorPalette[age_temp], linestyle = lineStyle[age_voltage], ax = axes[0])
        if age_temp==283.15:
            sns.lineplot(x = xdata, y = ydata, color = colorPalette[age_temp], linestyle = lineStyle[age_voltage], ax = axes[1])
        if age_temp==298.15:
            sns.lineplot(x = xdata, y = ydata, color = colorPalette[age_temp], linestyle = lineStyle[age_voltage], ax = axes[2])
        if age_temp==313.15:
            sns.lineplot(x = xdata, y = ydata, color = colorPalette[age_temp], linestyle = lineStyle[age_voltage], ax = axes[3])

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'age_voltage':age_voltage, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()

    start = [1,1,1,0.5]
    bounds = [(-5,5),(0,1),(-1000,1000),(0.2,0.8)]
    result = optimize.minimize(funktion1_min,x0 = start, bounds= bounds)
    print(result)
    print(xdata, ydata, age_temp, age_soc)
    # sns.lineplot(x = xdata, y = ydata, color = "red")
    
    temp = [273.15, 283.15, 298.15, 313.15]
    volt= [0.42, 2.1, 3.78, 4.2]
    for i_t in range(0, len(temp)):
        for i_v in range(0, len(volt)):
            age_voltage = volt[i_v]
            age_temp = temp[i_t]
            sns.lineplot(x = xdata, y = funktion1_expo(xdata, *result.x), color = "black", ax = axes[i_t], linestyle = lineStyle[age_voltage])

    plt.show()

def calculate_average_by_age_temp(data):
    data.sort(key=lambda x: x[1])  # Sort the data by age_temp
    grouped_data = {age_temp: list(values) for age_temp, values in groupby(data, key=lambda x: x[1])}

    # Calculate the average value for each group
    average_values = {}
    for age_temp, values in grouped_data.items():
        total_value = sum(value for _, _, value in values)
        average_value = total_value / len(values)
        average_values[age_temp] = average_value

    return average_values         

def lookuptable(arrayOfPathToCsvFiles, plotWithFixedExponent=False,fixedExponent=0.5):
    df_alleZellen = pd.DataFrame(columns=['nameCell', 'xdata', 'ydata', 'age_temp','age_soc'])
    df_popt = pd.DataFrame(columns=['nameCell', 'a0', 'age_temp', 'age_soc'])
    global age_temp, age_voltage, max_capacity,listOfPrograms,maxXValue,exponent
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStylePalette = {10:":",50:"--",90:"-.", 100:"-"}
  
# One programm is one temp and soc level  
    listOfPrograms = []
#   used to determine the max plot length to set an end to simulated data
    maxXValue = 0

    fig1 = plt.figure("cal_lookup_free_0C")
    fig2 = plt.figure("cal_lookup_free_10C")
    fig3 = plt.figure("cal_lookup_free_25C")
    fig4 = plt.figure("cal_lookup_free_40C")
    
    parameterSatzFig = {(273.15):fig1 ,
                    (283.15): fig2,
                    (298.15): fig3,
                    (313.15): fig4}



    # xdataForMean = []
    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]+273.15
        age_soc = df['age_soc'].iloc[0]
        #age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)
        
        if (plotWithFixedExponent):
            exponent = fixedExponent
        else:
            exponent=funktionExponentMitWerten(age_soc)
       

        relevant_df = df[['timestamp_s', 'cap_aged_est_Ah']].copy()
        # print(age_voltage, age_temp)
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




        p0 = [1]

        popt, pcov = optimize.curve_fit(funktion_lookuptable, xdata, ydata, p0=p0)#, bounds=(lower_bounds, upper_bounds))
        print("popt", popt)

        parameterSatz = (age_soc,age_temp)
        if parameterSatz not in listOfPrograms:
            listOfPrograms.append(parameterSatz)
        sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle= lineStylePalette[age_soc] ,ax = parameterSatzFig[(age_temp)].gca(), alpha = 0.5)
        
        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"

        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'a0': popt[0], 'age_temp':age_temp, 'age_soc': age_soc,}
            new_row2 = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'age_temp':age_temp, 'age_soc': age_soc}
        new_df = pd.DataFrame(new_row, index=[0])
        df_popt = pd.concat([df_popt, new_df], ignore_index=True)
        new_df2 = pd.DataFrame(new_row2, index=[0])
        df_alleZellen = pd.concat([df_alleZellen, new_df2], ignore_index=True)
    


    # subfigure.set_xlabel('timestamp_s')
    # subfigure.set_ylabel('cap_aged_est_Ah')
    # global_meanValues = df_popt.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).drop("age_voltage", axis=1).mean().values
    # print("df_popt",df_popt.sort_values("age_soc"))
    # print('global mean values', global_meanValues)
    uniqueZellen = df_alleZellen["nameCell"].unique()
    errorList = []
    for tuple in listOfPrograms:
        
        print("tuple",tuple)
        age_temp = tuple[1]
        age_soc = tuple[0]
        selected_rows = df_popt[(df_popt['age_temp'] == age_temp) & (df_popt['age_soc'] == age_soc)]

        #Calculate average values of all rows
        average_values = selected_rows.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("age_soc", axis=1).mean()
        print("local mean values", average_values.values)
        
        if (plotWithFixedExponent):
            exponent = fixedExponent
        else:
            exponent=funktionExponentMitWerten(age_soc)
        print("Exponent", exponent)
        # x_data = x_data/(60*60*24)
        x_data = np.linspace(0, maxXValue, 40)
        y_data = funktion_lookuptable(x_data, *average_values)
        sns.lineplot(x = x_data, y = y_data, linestyle= lineStylePalette[age_soc],  color = colorPalette[age_temp], ax = parameterSatzFig[age_temp].gca())
        
        for cell in uniqueZellen:
                if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_soc"].iloc[0] == age_soc) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0] == age_temp):
                    xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                    ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]

                    xdata = xdata[:len(ydata)]
                    
                    y_modellierte= funktion_lookuptable(xdata, *average_values)
                    error = (y_modellierte-ydata)**2
                    errorList.append((age_soc, age_temp, np.mean(error*1000)))
                    
            # print(f"RSME SOC age_voltage: {rmseParameterSatz}")
        


    errorDict = calculate_average_by_age_temp(errorList)
    print(errorDict)
    temp_list  = [273.15,283.15,298.15,313.15]
    for age_temp in temp_list:
        parameterSatz= (age_temp)
        line1 = plt.Line2D([0], [0], color="grey", lw=0, label ="SOC")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle=lineStylePalette[10], label ="10%")
        line3 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle=lineStylePalette[50], label ="50%")
        line4 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle=lineStylePalette[90], label ="90%")
        line5 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle=lineStylePalette[100],label ="100%")
        legend1 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')


        rmseString = "%.3f" % ((errorDict[age_temp])*1000)

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RSME: {rmseString}*e-3")
        legend2 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3], loc='upper right')
        parameterSatzFig[parameterSatz].gca().add_artist(legend1)
        parameterSatzFig[parameterSatz].gca().add_artist(legend2)


        parameterSatzFig[parameterSatz].gca().grid(color='lightgrey', linestyle='--')
        parameterSatzFig[parameterSatz].gca().set_xlabel(f'Alterung (Tage)')
        parameterSatzFig[parameterSatz].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        parameterSatzFig[parameterSatz].gca().set_title(f'Kalendarische LookUp Tabelle bei  {age_temp-273.15 }°C')


    plt.show()



        
   

    
















if __name__ == '__main__':

    # fit_mutiple([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_2_S16_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_2_S18_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_3_S19_C11.csv"])



    # plot_capacity_over_time(["all"])
    # fit_time_exponent(["all"])

    lookuptable([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_2_S16_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_2_S18_C11.csv"],False,0.5)
    # funktion6_plot_and_fit_overviewGreyAndRed([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_2_S16_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_2_S18_C11.csv"],plotTemperature=283.15,plotWithFixedExponent=True,fixedExponent=0.5)
    # funktion6_plot_and_fit_overviewGreyAndRed([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_2_S16_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_2_S18_C11.csv"],plotTemperature=298.15,plotWithFixedExponent=True,fixedExponent=0.5)
    # funktion6_plot_and_fit_overviewGreyAndRed([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_3_S19_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_1_S01_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_2_S04_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P001_3_S05_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_1_S02_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_2_S03_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P002_3_S04_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_1_S01_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_2_S02_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P003_3_S05_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_1_S03_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_2_S04_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P004_3_S05_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_1_S07_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_2_S08_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P005_3_S09_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_1_S06_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_2_S07_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P006_3_S08_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_1_S06_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_2_S09_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P007_3_S10_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_1_S07_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_2_S08_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P008_3_S09_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_1_S12_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_2_S13_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P009_3_S14_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_1_S11_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_2_S12_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P010_3_S13_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_1_S10_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_2_S11_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P011_3_S14_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_1_S12_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_2_S13_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P012_3_S14_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_1_S15_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_2_S16_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P013_3_S19_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_1_S17_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_2_S18_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P014_3_S19_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_1_S15_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_2_S16_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P015_3_S17_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_1_S15_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P016_2_S18_C11.csv"],plotTemperature=313.15,plotWithFixedExponent=True,fixedExponent=0.5)