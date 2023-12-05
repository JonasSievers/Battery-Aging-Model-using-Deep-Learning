# 1. Plot relative capacity over Ah entladen, over EFC ( and over time as info?)
# 2. Fit function 1 2 3
# cyc_condition
# cyc_charged
# age_temp
# age_chg_rate
# age_dischg_rate
# cap_aged_est_Ah
# total_q_dischg_Ah
# total_q_chg_Ah
# v_min_target_V
# v_max_target_V


import hashlib
import os
import re
import sys
import warnings
import seaborn as sns
import pandas as pd
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest
global dodCell,dodFactor, I_Charge, I_Discharge,cyc_temp, DOD,mid_SOC

np.set_printoptions(suppress=True)

# Global variables
max_capacity = 0
age_temp = 0
age_voltage = 0
exponent = 1
usePreGeneratedFiles = True
dsoc=1

# DEFINE PATHS TO CSV FILES
filepathToDirectory = "C:\\Users\\Yannick\\bwSyncShare\\MA Yannick Fritsch\\00_Daten\\prepr_res_run_7\\"
tempFilesDirectory = "C:\\Users\\Yannick\\Desktop\\tempSkripte\\"
exponentFile = "cyc_exponent.csv"
## HELPER FUNCTION FOR PLOTTING

arrayPackagesToDisplay = ["eoc",]
arrayCalendarParameterID = ["P001", "P002", "P003", "P004", "P005", "P006","P007", "P008", "P009", "P010", "P011", "P012", "P013", "P014", "P015", "P016"]
arrayCyclicParameterID = ["P017", "P018", "P019", "P020", "P021", "P022", "P023", "P024", "P025", "P026", "P027", "P028", "P029", "P030", "P031", "P032", "P033", "P034", "P035", "P036", "P037", "P038", "P039", "P040", "P041", "P042", "P043", "P044", "P045", "P046", "P047", "P048", "P048", "P049", "P050", "P051", "P052", "P053", "P054", "P055", "P056", "P057", "P058", "P059", "P060", "P061", "P062", "P063", "P064"]
arrayProfileParameterID = ["P065", "P066", "P067", "P068","P069", "P070", "P071", "P072", "P073", "P074", "P075", "P076"]
ARRAY_CELLS_TO_DISPLAY = arrayCyclicParameterID


def funktion7_expo(x,a0,a1,a2,a3,a4):
    return 1 - (a0*np.exp(a1*dsoc) +a2*np.exp(a3*dsoc))*np.power(x,a4)
def funktion7(x,a0,a1,a2,a3):
    return 1 - (a0*np.exp(a1*dsoc) +a2*np.exp(a3*dsoc)*np.power(x,exponent))
def funktion8(x,a0,a1):
    ce = faktorCEBerechnen(x)
    return 1- a0*np.power(ce,x)+a1
def funktion9(x,a0,a1,):
    return 1- faktorACBerechnen(mid_SOC)*np.exp(a0/age_temp)*np.power(x,a1)
def faktorACBerechnen(SOC):
    return SOC
def faktorCEBerechnen(x):
    return True


# def MatFunc(x,a0,a1,a2,a3,a4,a5,a6,a7):
#     return ((a0+a1*I_Charge*a2*I_Charge*I_Charge)*(a3+a4*I_Discharge+a5*I_Discharge*I_Discharge)*a6*np.exp(a6/cyc_temp)*np.power(x,a7))
def helper_df_selected_parameters(filesToSearch: list):
    global usePreGeneratedFiles
    tempFileName ="cyc_selected_parameters.csv"
    files = os.listdir(filepathToDirectory)
    files = [f for f in files if any(substring in f for substring in filesToSearch)]
    files.sort()
    data = {
        'fileName': [],
        'filePath': [],
        'SlaveNumber': [],
        'CellNumber': [],
        "parameterId" : [],
        "v_max_target_V": [],
        "v_min_target_V": [],
        "age_temp":[],
        "age_dischg_rate":[],
        "age_chg_rate":[]

    }
    df = pd.DataFrame(data)
    #Fill the dataframe with info of csv files
    if os.path.exists(tempFilesDirectory+tempFileName) & usePreGeneratedFiles:
        print(f"A file with the name '{tempFileName}' already exists in the folder. Using that data instead!")
        df = pd.read_csv(tempFilesDirectory+tempFileName,sep=";",header=0)
    else:
        print(f"No file with the name '{tempFileName}' exists in the folder. Calculating data...")

        for file in files:
            # Extract PXXX
            p_number = re.search(r'P(\d+)', file).group(1)
            # Extract SXX
            s_number = re.search(r'S(\d+)', file).group(1)
            # Extract CXX
            c_number = re.search(r'C(\d+)', file).group(1)
            temp_df = pd.read_csv(filepathToDirectory+file,usecols=["cyc_condition","cyc_charged","v_max_target_V","v_min_target_V","age_temp","age_dischg_rate","age_chg_rate"],sep=";",header =0, nrows=500)
            temp_df = temp_df[(temp_df["cyc_condition"]==1) &(temp_df["cyc_charged"]==1)]
            v_max_target_V = temp_df["v_max_target_V"].iloc[0]
            v_min_target_V = temp_df["v_min_target_V"].iloc[0]
            age_temp = temp_df["age_temp"].iloc[0]
            age_dischg_rate = temp_df["age_dischg_rate"].iloc[0]
            age_chg_rate = temp_df["age_chg_rate"].iloc[0]
            print(f"File {file}, v_max_target_V {v_max_target_V}, v_min_target_V {v_min_target_V}")






            file_data = {
                'fileName': file,
                'filePath': filepathToDirectory+file,
                'SlaveNumber': s_number,
                'CellNumber': c_number,
                "parameterId" : p_number,
                "v_max_target_V": v_max_target_V,
                "v_min_target_V": v_min_target_V,
                "age_temp": age_temp,
                "age_dischg_rate": age_dischg_rate,
                "age_chg_rate":age_chg_rate
            }
            df = pd.concat([df, pd.DataFrame(file_data, index=[0])], ignore_index=True)
        if (not usePreGeneratedFiles):
            df.to_csv(tempFilesDirectory+tempFileName, sep =';', header= True, mode="w")
            print(f"DataFrame saved to '{tempFileName}' in the folder{tempFilesDirectory}.")
    return df
def calculate_checksum(data):
    hash_object = hashlib.md5(str(data).encode())
    return hash_object.hexdigest()
def plot_and_fit_MatFunc(pathToCsvFile: str):
    global age_temp, age_voltage, max_capacity,cyc_temp,I_Charge,I_Discharge
    df = pd.read_csv(
        pathToCsvFile, header=0, delimiter=";")


    df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2) & (df['cyc_condition'] == 2)]
    age_temp = df['age_temp'].iloc[0]+273.15
    age_soc = df['age_soc'].iloc[0]
    I_Charge = df['age_chg_rate'].iloc[0]
    I_Discharge = df['age_dischg_rate'].iloc[0]
    age_voltage = age_soc * 4.2 / 100
    cyc_temp = float(age_temp)
    relevant_df = df[['total_q_chg_Ah', 'cap_aged_est_Ah']]
    # relevant_df.loc[:, 'total_q_chg_Ah'] = relevant_df['total_q_chg_Ah'] - relevant_df['timestamp_s'].iloc[0]
    relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
    max_capacity = df['cap_aged_est_Ah'].iloc[0]
    relevant_df = relevant_df.reset_index()
    xdata = relevant_df['total_q_chg_Ah']
    ydata = relevant_df['cap_aged_est_Ah']

    # Set the style of the plot
    sns.set_style("darkgrid")

    # Plot the data
    sns.lineplot(data=relevant_df, x="total_q_chg_Ah", y="cap_aged_est_Ah", marker="o")
    p0 = [1,1,1,1,1,1,1,1]
    # lower_bounds = [-5, -5, -5, -1000]
    # upper_bounds = [5, 5, 5, 100]
    # , bounds=(lower_bounds, upper_bounds)
    popt, pcov = optimize.curve_fit(MatFunc, xdata, ydata, p0=p0)
    print(popt)
    sns.lineplot(x=xdata, y=MatFunc(xdata, *popt), marker="v")

    # Set labels and title
    plt.xlabel("total_q_chg_Ah (Ah)")
    plt.ylabel("Cap Aged Estimated (Ah)")
    plt.title("Plot of Timestamp vs Cap Aged Estimated")

    # Display the plot
    plt.show()
def custom_max(values):
    return max(value for value in values if value is not None)
def custom_min(values):
    return min(value for value in values if value is not None)



def plot_capacity_over_time(arrayOfPathsToCsvFiles: list):
    global usePreGeneratedFiles
    # Plot limits:
    max_capacity = 1
    min_capacity = 0.75
    # Name of tempFile in the temp directory
    tempFileName = "cyc_plot_capacity_over_time.csv"
    # Generated list of PXXX of the cells
    listOfPrograms = []


    df_cells = pd.DataFrame()
    df_cell_csv_content = pd.DataFrame()


    if len(arrayOfPathsToCsvFiles) == 0:
        print("No csv files to plot")
        return False
    if (len(arrayOfPathsToCsvFiles) == 1) and (arrayOfPathsToCsvFiles[0] == "all"):
        df_cells = helper_df_selected_parameters(ARRAY_CELLS_TO_DISPLAY)
        print(arrayOfPathsToCsvFiles)
    else:
        df_cells = helper_df_selected_parameters(arrayOfPathsToCsvFiles)

    print(df_cells.head())
    
    if os.path.exists(tempFilesDirectory+tempFileName) & usePreGeneratedFiles:
        print(f"A file with the name '{tempFileName}' already exists in the folder. Using that data instead!")
        df_cell_csv_content = pd.read_csv(tempFilesDirectory+tempFileName, sep=";",header=0)
    else:
        for index,row in df_cells.iterrows():
            df = pd.read_csv(row["filePath"], sep=";", header=0)
            df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
            df.loc[:, 'cap_aged_est_Ah'] = df['cap_aged_est_Ah'] / df['cap_aged_est_Ah'].iloc[0]
            df["cap_aged_est_Ah"]
            df["fileName"] = row["fileName"]
            df["v_max_target_V"]=row["v_max_target_V"]
            df["v_min_target_V"]=row["v_min_target_V"]
            df_cell_csv_content = pd.concat([df_cell_csv_content,df],ignore_index=True)
        if (not usePreGeneratedFiles):
            df_cell_csv_content.to_csv(tempFilesDirectory+tempFileName, index=False, sep=";", header=True)
            print(f"DataFrame saved to '{tempFileName}' in the folder{tempFilesDirectory}.")
    print(df_cell_csv_content)
    colorPalette = {0.0:"blue",10: "green",25: "orange", 40: "red"}
    uniqueFiles= df_cell_csv_content["fileName"].unique()

    

    for file in uniqueFiles:
        df = df_cell_csv_content[df_cell_csv_content["fileName"] == file].reset_index()
        colorInt = df["age_temp"][0]
        # lineStyleInt = df["age_soc"][0]
        row = df_cells[df_cells["fileName"]==file].iloc[0]
        tuple_to_check = (row["v_max_target_V"],row["v_min_target_V"] ,row["age_chg_rate"],row["age_dischg_rate"])
        if tuple_to_check not in listOfPrograms:
            print(f"Tuple not found in the list, adding {tuple_to_check}")
            listOfPrograms.append(tuple_to_check)
        sns.lineplot(data=df, x="total_q_chg_Ah", y = "cap_aged_est_Ah", color = colorPalette[colorInt]) #linestyle = lineStyle[lineStyleInt] )
    print("All tuples are:",listOfPrograms)
    plt.xlabel("Gesamte Ladung (Ah)")
    plt.ylabel("Rel. Kapazität")
    plt.title("Rel. Kapazität (Ah) im Verlauf der Alterung")
    
    # Display the plot
    plt.show()
    

    
    uniqueSoc = df_cell_csv_content["age_soc"].unique()
    uniqueTemp= df_cell_csv_content["age_temp"].unique()
    
    
    


    # Für jedes Tuple gibt es ein Plot
    for tuple in listOfPrograms:
        fig, ax = plt.subplots(ncols=1, nrows=1)

        for temp in uniqueTemp:
            print(f'-------Tuple: {tuple}, temp: {temp}--------')

            # select the correct part of the df with one temp
            df_tuple= df_cell_csv_content[(df_cell_csv_content["v_max_target_V"]==tuple[0]) &(df_cell_csv_content["v_min_target_V"]==tuple[1])&(df_cell_csv_content["age_chg_rate"]==tuple[2])&(df_cell_csv_content["age_dischg_rate"]==tuple[3])&(df_cell_csv_content["age_temp"]==temp)]
            df_tuple = df_tuple[['total_q_dischg_Ah','timestamp_s', 'cap_aged_est_Ah',"fileName"]]
            uniqueFiles = df_tuple["fileName"].unique()
            y_values =[]
            for uniqueFile in uniqueFiles:
                y_values.append(df_tuple[df_tuple["fileName"] == uniqueFile]["cap_aged_est_Ah"].values)
            maxcap_aged_est_Ah=int(df_tuple["total_q_dischg_Ah"].max())
            maxTimestamp = int(df_tuple["timestamp_s"].max())
            
            min_values = []
            max_values = []
            mean_values = []
            print(*y_values)
            

            #  NEW METHOD - Sonst ist sind die arrays so kurz wie die kürzesten Daten!
            # max_values = [custom_max(values) for values in zip_longest(*y_values, fillvalue=None)]
            # min_values = [custom_min(values) for values in zip_longest(*y_values, fillvalue=None)]
            # mean_values = [sum(filter(None, values)) / len(list(filter(None, values))) for values in zip_longest(*y_values, fillvalue=None)]
            # NICHT MEHR BENUTZEN, verliert Daten wenn ein array kürzer ist!
            for values in zip(*y_values):
                print(values)
                min_values.append(min(values))
                max_values.append(max(values))
                mean_values.append(sum(values) / len(values))

            print(min_values)
            print(max_values)
            print(f'Tuple: {tuple}, temp: {temp}')
            print(mean_values)
            # CALCULATE RMSE
            # print("mean",np.mean(y_values, axis=0))
            # mean = np.mean(y_values, axis=0)
            # print("standard deviation",np.std(y_values, axis=0))
            # print("Mean of standard deviation",np.mean(np.std(y_values, axis=0), axis=0))
            # print("rsme",np.sqrt(np.mean((mean-y_values)**2)))           
            # x_values = np.array([i for i in range(0,(maxTimestamp-minTimestamp), int((maxTimestamp-minTimestamp)/(len(min_values))))])
            x_values = np.array(np.linspace(0, (maxcap_aged_est_Ah), len(mean_values), dtype=int))
            # print(x_values)
            x_values = x_values/(3)
            # print(len(min_values),len(max_values),len(mean_values),len(x_values))

            ax.plot(x_values, mean_values,color=colorPalette[temp], label=f'{temp}°C')
            ax.fill_between(x_values, min_values, max_values, interpolate = True, alpha = 0.05, color=colorPalette[temp])
        text1=""
        if tuple[1] == 2.5:
            text1 = "0-100%"
        elif tuple[0]==4.092:
            text1 = "10-90%"
        else:
            text1 = "10-100%"
        
        text2 =""
        match tuple[2]:
            case 0.33:
                text2 = "1/3C"
            case 1:
                text2 = "1C"
            case 1.67:
                text2 = "5/3C"
        text3=""
        match tuple[3]:
            case 0.33:
                text3 = "1/3C"
            case 1:
                text3 = "1C"
            
        plt.title(f"{text1}-Zyklen Laden: {text2} Entladen: {text3}")
        plt.xlabel('Anzahl an EFC (-)')
        plt.ylabel('Relative Kapazität C(t)/C_init (-)')
        plt.grid(color='lightgrey', linestyle='--')
        
        plt.ylim(min_capacity, max_capacity)
        plt.xlim(0, 1750)
        plt.legend()
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        plt.show()
# -------------------------------------------------------------------------------------------------------------------------------------------

def fit_exponent(arrayOfPathsToCsvFiles: list):
    global usePreGeneratedFiles,dsoc, exponentFile
    # Plot limits:
    max_capacity = 1
    min_capacity = 0.75
    # Name of tempFile in the temp directory
    tempFileName = "cyc_plot_capacity_over_time.csv"
    
    # Generated list of PXXX of the cells
    listOfPrograms = []


    df_cells = pd.DataFrame()
    df_cell_csv_content = pd.DataFrame()
    df_cells_exponent = pd.DataFrame()


    if len(arrayOfPathsToCsvFiles) == 0:
        print("No csv files to plot")
        return False
    if (len(arrayOfPathsToCsvFiles) == 1) and (arrayOfPathsToCsvFiles[0] == "all"):
        df_cells = helper_df_selected_parameters(ARRAY_CELLS_TO_DISPLAY)
        print(arrayOfPathsToCsvFiles)
    else:
        df_cells = helper_df_selected_parameters(arrayOfPathsToCsvFiles)

    print(df_cells.head())
    
    if os.path.exists(tempFilesDirectory+tempFileName) & usePreGeneratedFiles:
        print(f"A file with the name '{tempFileName}' already exists in the folder. Using that data instead!")
        df_cell_csv_content = pd.read_csv(tempFilesDirectory+tempFileName, sep=";",header=0)
    else:
        for index,row in df_cells.iterrows():
            df = pd.read_csv(row["filePath"], sep=";", header=0)
            df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2) ]
            df.loc[:, 'cap_aged_est_Ah'] = df['cap_aged_est_Ah'] / df['cap_aged_est_Ah'].iloc[0]
            df["cap_aged_est_Ah"]
            df["fileName"] = row["fileName"]
            df["v_max_target_V"]=row["v_max_target_V"]
            df["v_min_target_V"]=row["v_min_target_V"]
            df["dsoc"]= row["v_max_target_V"] -row["v_min_target_V"]
            df_cell_csv_content = pd.concat([df_cell_csv_content,df],ignore_index=True)
        if (not usePreGeneratedFiles):
            df_cell_csv_content.to_csv(tempFilesDirectory+tempFileName, index=False, sep=";", header=True)
            print(f"DataFrame saved to '{tempFileName}' in the folder{tempFilesDirectory}.")
    print(df_cell_csv_content)
    colorPalette = {0.0:"blue",10: "green",25: "orange", 40: "red"}
    uniqueFiles= df_cell_csv_content["fileName"].unique()

    # DSOC = maxV - minV -> DSOC = v_max_target_V-v_min_target_V
    # Temperature
    # Charge speed 

    for file in uniqueFiles:
        print(f"--------{file}---------")
        df = df_cell_csv_content[df_cell_csv_content["fileName"] == file].reset_index()
        row_df_cells = df_cells[df_cells["fileName"]==file].iloc[0]
        dsoc=row_df_cells["v_max_target_V"]-row_df_cells["v_min_target_V"]
        print(f"DSOC: {dsoc}")
        age_temp = df['age_temp'].iloc[0]+273.15
        age_temp = float(age_temp)
        currentCharge = row_df_cells["age_chg_rate"]
        df = df[['total_q_dischg_Ah','timestamp_s', 'cap_aged_est_Ah',"fileName"]]

        df.loc[:, 'cap_aged_est_Ah'] = df['cap_aged_est_Ah'] / df['cap_aged_est_Ah'].iloc[0]
        df = df.reset_index()
        # Load the data from the pandas dataframe
        xdata = df['total_q_dischg_Ah'].values
        ydata = df['cap_aged_est_Ah'].values
        
        # If there are less than 6 points of data - interpolate it to have 6 points
        if len(xdata)<10:
            print(f"Not enough points length is {len(xdata)}, need to interpolate")
            # Number of points in interpolation
            num_points = 10

            # Indices for the original array
            xInd = np.arange(len(xdata))
            yInd = np.arange(len(ydata))

            # Indices for the new interpolated array
            new_x = np.linspace(0, len(xdata) - 1, num_points)
            new_y = np.linspace(0, len(ydata) - 1, num_points)

            # Perform linear interpolation
            new_xdata = np.interp(new_x, xInd, xdata)
            new_ydata = np.interp(new_y, yInd, ydata)
            xdata = new_xdata
            ydata = new_ydata
        print(f"xdata: {xdata} \n ydata:  {ydata}")

        # Fit the FUNKTION1 to the data and extract the time exponent
        p0 = [1,1,1,1,0.6]
        lower_bounds = [-5, -5, -5, -5, 0]
        upper_bounds = [5, 5, 5, 5, 1]
        popt, pcov = optimize.curve_fit(funktion7_expo, xdata, ydata, p0=p0,bounds=[lower_bounds, upper_bounds])
        temp_df = pd.DataFrame({"fileName":file, "fitFunction": "funktion1", "exponent":popt[4], "age_temp": age_temp, "dsoc":dsoc, "currentCharge":currentCharge}, index=[0])
        df_cells_exponent = pd.concat([df_cells_exponent,temp_df],ignore_index=True)
        print(f"Parameters {popt}")
        
        # print(df_cells_exponent.head())
        # if currentCharge == 1.67:
        #     sns.lineplot(x=xdata, y =ydata, color = "blue")
        #     sns.lineplot(x=xdata, y =funktion1(xdata,*popt), color ="red") 
        #     plt.show()
    print(df_cells_exponent)
    df_forboxplot = df_cells_exponent#[(df_cells_exponent["exponent"]>0.5)]
    averages =df_forboxplot.groupby(['currentCharge','dsoc','age_temp'])["exponent"].mean()
    print(averages)
    averages.to_csv(tempFilesDirectory+exponentFile, sep=";",header=True)
    return True
    print(averages.index.values, type(averages.index.values))
    print(averages.values, type(averages.values))
    xdata=averages.index.values.astype(float)
    ydata=averages.values.astype(float)
    x_data2 = np.linspace(xdata[0], xdata[-1], 50)
    sns.boxplot(data=df_forboxplot, x="currentCharge", y = "exponent", width = 0.5)
    print(f"Average values: {ydata}")
    df_corr = df_forboxplot [["exponent","age_temp","dsoc","currentCharge"]]
    print(df_corr.corr())
    
    plt.show()
    
    return True
    sns.lineplot(x = x_data2,y = funktionExponent(x_data2,*popt))
    plt.show()  
    p0 = [1,1,1]
    popt, pcov = optimize.curve_fit(funktionExponent, xdata, ydata, p0=p0)
    print(x_data2, type(x_data2))
    print(f"Modell für Exponent: {popt[0]}+{popt[1]}*x+{popt[2]}*x^2")
    sns.boxplot(data=df_forboxplot, x="age_soc", y = "exponent", width = 0.5) 
    plt.show()
    



def funktion7_plot_and_fit_overviewGreyAndRed(arrayOfPathsToCsvFiles):
    global usePreGeneratedFiles,exponentFile,exponent,dsoc
    # Plot limits:
    max_capacity = 1
    min_capacity = 0.75
    # Name of tempFile in the temp directory
    tempFileName = "cyc_plot_capacity_over_time.csv"
    # Generated list of PXXX of the cells
    listOfPrograms = []


    df_cells = pd.DataFrame()
    df_cell_csv_content = pd.DataFrame()
    df_popt = pd.DataFrame(columns=['nameCell','a0', 'a1', 'a2','a3', 'age_temp', 'dsoc', 'currentCharge'])
    lineStyle = {273.15:":",283.15:"-.",298.15:"-", 313.15:"--"}
    colorPalette = {0.0:"blue",10: "green",25: "orange", 40: "red"}


    if len(arrayOfPathsToCsvFiles) == 0:
        print("No csv files to plot")
        return False
    if (len(arrayOfPathsToCsvFiles) == 1) and (arrayOfPathsToCsvFiles[0] == "all"):
        df_cells = helper_df_selected_parameters(ARRAY_CELLS_TO_DISPLAY)
        print(arrayOfPathsToCsvFiles)
    else:
        df_cells = helper_df_selected_parameters(arrayOfPathsToCsvFiles)

    print(df_cells.head())
    
    if os.path.exists(tempFilesDirectory+tempFileName) & usePreGeneratedFiles:
        print(f"A file with the name '{tempFileName}' already exists in the folder. Using that data instead!")
        df_cell_csv_content = pd.read_csv(tempFilesDirectory+tempFileName, sep=";",header=0)
    else:
        for index,row in df_cells.iterrows():
            df = pd.read_csv(row["filePath"], sep=";", header=0)
            df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
            df.loc[:, 'cap_aged_est_Ah'] = df['cap_aged_est_Ah'] / df['cap_aged_est_Ah'].iloc[0]
            df["cap_aged_est_Ah"]
            df["fileName"] = row["fileName"]
            df["v_max_target_V"]=row["v_max_target_V"]
            df["v_min_target_V"]=row["v_min_target_V"]
            df_cell_csv_content = pd.concat([df_cell_csv_content,df],ignore_index=True)
        if (not usePreGeneratedFiles):
            df_cell_csv_content.to_csv(tempFilesDirectory+tempFileName, index=False, sep=";", header=True)
            print(f"DataFrame saved to '{tempFileName}' in the folder{tempFilesDirectory}.")
    print(df_cell_csv_content)
    uniqueFiles= df_cell_csv_content["fileName"].unique()

    

    for file in uniqueFiles:
        df = df_cell_csv_content[df_cell_csv_content["fileName"] == file].reset_index()
        colorInt = df["age_temp"][0]
        # lineStyleInt = df["age_soc"][0]
        row = df_cells[df_cells["fileName"]==file].iloc[0]
        tuple_to_check = (row["v_max_target_V"],row["v_min_target_V"] ,row["age_chg_rate"],row["age_dischg_rate"])
        if tuple_to_check not in listOfPrograms:
            print(f"Tuple not found in the list, adding {tuple_to_check}")
            listOfPrograms.append(tuple_to_check)
    print("All tuples are:",listOfPrograms)
    uniqueSoc = df_cell_csv_content["age_soc"].unique()
    uniqueTemp= df_cell_csv_content["age_temp"].unique()

    # Open Exponent file
    df_exponent = pd.read_csv(tempFilesDirectory+exponentFile, sep=";",header=0)
    print(df_exponent.head())

    # Für jedes Tuple gibt es ein Plot
    for tuple in listOfPrograms:
        fig, ax = plt.subplots(ncols=1, nrows=1)

        for temp in uniqueTemp:
            print(f'-------Tuple: {tuple}, temp: {temp}--------')

            # select the correct part of the df with one temp
            df_tuple= df_cell_csv_content[(df_cell_csv_content["v_max_target_V"]==tuple[0]) &(df_cell_csv_content["v_min_target_V"]==tuple[1])&(df_cell_csv_content["age_chg_rate"]==tuple[2])&(df_cell_csv_content["age_dischg_rate"]==tuple[3])&(df_cell_csv_content["age_temp"]==temp)]
            df_tuple = df_tuple[['total_q_dischg_Ah','timestamp_s', 'cap_aged_est_Ah',"fileName","age_temp"]]
            uniqueFiles = df_tuple["fileName"].unique()
            y_values =[]
            for uniqueFile in uniqueFiles:
                y_values = df_tuple[df_tuple["fileName"] == uniqueFile]["cap_aged_est_Ah"].values
                age_temp = df_tuple[df_tuple["fileName"] == uniqueFile]["age_temp"].iloc[0] + 273.15
                maxcap_aged_est_Ah=int(df_tuple[df_tuple["fileName"] == uniqueFile]["total_q_dischg_Ah"].max())
                x_values = np.array(np.linspace(0, (maxcap_aged_est_Ah), len(y_values), dtype=int))
                x_values = x_values/(3) #Transform from Ah to EFC
                p0 = [5,1,1,2]
                lower_bounds = [-100, -100, -100, -100]
                upper_bounds = [50, 50, 50, 50]
                ax.plot(x_values, y_values,color="grey",linestyle=lineStyle[age_temp])
                popt, pcov = optimize.curve_fit(funktion7, x_values, y_values, p0=p0,bounds=[lower_bounds, upper_bounds])
                print("popt", popt)

                pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"

                match = re.search(pattern, uniqueFile)
                if match:
                    extracted_value = match.group()
                    # ax.set_title(extracted_value)
                    new_row = {'nameCell': extracted_value,
                            'a0': popt[0], 'a1': popt[1], 'a2': popt[2],'a3': popt[3], 'age_temp':temp, 'dsoc':dsoc, 'currentCharge':tuple[3]}
                
                new_df = pd.DataFrame(new_row, index=[0])
                df_popt = pd.concat([df_popt, new_df], ignore_index=True)

            selected_rows = df_popt[(df_popt['age_temp'] == temp) & (df_popt['dsoc'] == dsoc)&(df_popt['currentCharge'] == tuple[3])]

            #Calculate average values of all rows
            average_values = selected_rows.drop("nameCell", axis=1).drop("age_temp", axis=1).drop("dsoc", axis=1).drop("currentCharge", axis=1).mean()
            exponent = df_exponent[(df_exponent["currentCharge"]==tuple[3])&(df_exponent["dsoc"]==tuple[0]-tuple[1])&(df_exponent["age_temp"]==temp+273.15)]["exponent"].iloc[0]
            print(f"Exponent {exponent}")
            dsoc =tuple[0]-tuple[1]
            x_data = np.linspace(0,1750,20)
            y_data = funktion7(x_data,*average_values)
            ax.plot(x_data, y_data , color="red",linestyle=lineStyle[age_temp])



            
        text1=""
        if tuple[1] == 2.5:
            text1 = "0-100%"
        elif tuple[0]==4.092:
            text1 = "10-90%"
        else:
            text1 = "10-100%"
        
        text2 =""
        match tuple[2]:
            case 0.33:
                text2 = "1/3C"
            case 1:
                text2 = "1C"
            case 1.67:
                text2 = "5/3C"
        text3=""
        match tuple[3]:
            case 0.33:
                text3 = "1/3C"
            case 1:
                text3 = "1C"
        line1 = plt.Line2D([0], [0], color="grey", lw=0, label ="Temperatur °K")
        line2 = plt.Line2D([0], [0], color="grey", lw=2,linestyle=':', label ="273.15")
        line3 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-.', label ="283.15")
        line4 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-', label ="298.15")
        line5 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='--',label ="313.15")
        legend1 = plt.legend(handles=[line1, line2, line3,line4,line5], loc='upper right')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color="red", lw=2, label = "Modellierung")
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RSME: %")
        legend2 = plt.legend(handles=[line1, line2, line3], loc='lower right')
        plt.gca().add_artist(legend1)
        plt.gca().add_artist(legend2)
        plt.title(f"{text1}-Zyklen Laden: {text2} Entladen: {text3}")
        plt.xlabel('Anzahl an EFC (-)')
        plt.ylabel('Relative Kapazität C(t)/C_init (-)')
        plt.grid(color='lightgrey', linestyle='--')
        
        plt.ylim(min_capacity, max_capacity)
        plt.xlim(0, 1750)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        plt.show()



        






    
def plot_and_fit_mutiple(arrayOfPathToCsvFiles):
    global age_temp, age_voltage, max_capacity
    # show the right number of columns
    # Set the style of the plot
    sns.set_style("darkgrid")

    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2) & (df['cyc_condition'] == 2)]
        age_temp = df['age_temp'].iloc[0]
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)

        relevant_df = df[['total_q_chg_Ah', 'cap_aged_est_Ah']]
        # relevant_df.loc[:, 'total_q_chg_Ah'] = relevant_df['total_q_chg_Ah'] - relevant_df['timestamp_s'].iloc[0]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        xdata = relevant_df['total_q_chg_Ah']
        ydata = relevant_df['cap_aged_est_Ah']

        
        # Plot the data
        sns.lineplot( x=xdata, y=ydata, marker="o")

    # Adjust the layout
    # plt.tight_layout()
    # Set the title of the axis
    plt.axhline(y=0.7, color='red', linestyle='--')

    plt.xlabel('total_q_chg_Ah')
    plt.ylabel('cap_aged_est_Ah')
    plt.legend()
    plt.show()

def plot_and_fit_mutiplev2(arrayOfPathToCsvFiles):
    total_df = pd.DataFrame()

    for file_name in arrayOfPathToCsvFiles:
        p_number = re.search(r'P(\d+)', file_name).group(1)
        p_x_number = re.search(r"P(\d+_\d+)", file_name).group(1)
        df = pd.read_csv(file_name, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2) & (df['cyc_condition'] == 2)]
        df['parameterId'] = p_number
        df['cellId'] = p_x_number
        total_df = pd.concat([total_df,df], ignore_index=True)

    total_df = total_df[(total_df['parameterId'] == "017")]
    print(total_df)
    sns.set_style("darkgrid")
    sns.lineplot(data = total_df ,x="total_q_chg_Ah", y="cap_aged_est_Ah", hue = "parameterId" )
    plt.axhline(y=0.7*3, color='red', linestyle='--')
    plt.xlabel('total_q_chg_Ah')
    plt.ylabel('cap_aged_est_Ah')
    plt.legend()
    plt.show()
    fmri = sns.load_dataset("fmri")
    print(fmri)

def plot_and_fit_mutiple_cutoff(arrayOfPathToCsvFiles):
    global age_temp, age_voltage, max_capacity
    # show the right number of columns
    # Set the style of the plot
    sns.set_style("darkgrid")

    for pathToCell in arrayOfPathToCsvFiles:
        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2) & (df['cyc_condition'] == 2)]

        age_temp = df['age_temp'].iloc[0]
        age_soc = df['age_soc'].iloc[0]
        age_voltage = age_soc * 4.2 / 100
        age_temp = float(age_temp)

        relevant_df = df[['total_q_chg_Ah', 'cap_aged_est_Ah']]
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]

        num_rows = len(relevant_df.index)
        # Check if there are more than 3 rows in the DataFrame
        if num_rows > 3:
            # Filter rows based on the condition and create a new DataFrame
            relevant_df = relevant_df.drop(relevant_df[relevant_df['cap_aged_est_Ah'] < 0.7].index)

        max_capacity = df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        xdata = relevant_df['total_q_chg_Ah']
        ydata = relevant_df['cap_aged_est_Ah']

        
        # Plot the data
        sns.lineplot( x=xdata, y=ydata, marker="o")

    # Adjust the layout
    # plt.tight_layout()
    # Set the title of the axis
    plt.axhline(y=0.7, color='red', linestyle='--')

    plt.xlabel('total_q_chg_Ah')
    plt.ylabel('cap_aged_est_Ah')
    plt.legend()
    plt.show()

def flatten_2d_array(arr):
    return np.array(arr).flatten()


def plot_and_fit(pathToCsvFile: str):
    global age_temp, age_voltage, max_capacity
    df = pd.read_csv(
        pathToCsvFile, header=0, delimiter=";")


    df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2) & (df['cyc_condition'] == 2)]
    age_temp = df['age_temp'].iloc[0]
    age_soc = df['age_soc'].iloc[0]
    age_voltage = age_soc * 4.2 / 100
    age_temp = float(age_temp)

    relevant_df = df[['total_q_chg_Ah', 'cap_aged_est_Ah']]
    # relevant_df.loc[:, 'total_q_chg_Ah'] = relevant_df['total_q_chg_Ah'] - relevant_df['timestamp_s'].iloc[0]
    relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
    max_capacity = df['cap_aged_est_Ah'].iloc[0]
    relevant_df = relevant_df.reset_index()
    xdata = relevant_df['total_q_chg_Ah']
    ydata = relevant_df['cap_aged_est_Ah']

    # Set the style of the plot
    sns.set_style("darkgrid")

    # Plot the data
    sns.lineplot(data=relevant_df, x="total_q_chg_Ah", y="cap_aged_est_Ah", marker="o")
    p0 = [1,1,1,1,1,1,1,1]
    # lower_bounds = [-5, -5, -5, -1000]
    # upper_bounds = [5, 5, 5, 100]
    # , bounds=(lower_bounds, upper_bounds)
    popt, pcov = optimize.curve_fit(func, xdata, ydata, p0=p0)
    print(popt)
    sns.lineplot(x=xdata, y=func(xdata, *popt), marker="v")

    # Set labels and title
    plt.xlabel("total_q_chg_Ah (Ah)")
    plt.ylabel("Cap Aged Estimated (Ah)")
    plt.title("Plot of Timestamp vs Cap Aged Estimated")

    # Display the plot
    plt.show()

if __name__ == '__main__':
    # funktion7_plot_and_fit_overviewGreyAndRed(['all'])
    plot_capacity_over_time(['all'])
    # fit_exponent(['all'])
    # plot_and_fit_MatFunc(r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P017_1_S01_C08.csv")
    # funktion7_plot_and_fit_overviewGreyAndRed([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P064_2_S18_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P064_3_S19_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P017_1_S01_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P017_2_S04_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P017_3_S05_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P018_1_S01_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P018_2_S02_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P018_3_S04_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P019_1_S01_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P019_2_S02_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P019_3_S05_C02.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P020_1_S01_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P020_2_S02_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P020_3_S03_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P021_1_S02_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P021_2_S03_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P021_3_S04_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P022_1_S03_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P022_2_S04_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P022_3_S05_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P023_1_S03_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P023_2_S04_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P023_3_S05_C03.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P024_1_S01_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P024_2_S04_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P024_3_S05_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P025_1_S01_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P025_2_S02_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P025_3_S03_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P026_1_S01_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P026_2_S02_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P026_3_S03_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P027_1_S01_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P027_2_S02_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P027_3_S03_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P028_1_S02_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P028_2_S03_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P028_3_S04_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P029_1_S06_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P029_2_S08_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P029_3_S09_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P030_1_S06_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P030_2_S08_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P030_3_S09_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P031_1_S06_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P031_2_S07_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P031_3_S09_C03.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P032_1_S06_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P032_2_S07_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P032_3_S08_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P033_1_S07_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P033_2_S08_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P033_3_S09_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P034_1_S07_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P034_2_S08_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P034_3_S09_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P035_1_S08_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P035_2_S09_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P035_3_S10_C02.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P036_1_S05_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P036_2_S09_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P036_3_S10_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P037_1_S05_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P037_2_S06_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P037_3_S10_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P038_1_S06_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P038_2_S07_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P038_3_S10_C03.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P039_1_S05_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P039_2_S06_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P039_3_S07_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P040_1_S06_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P040_2_S07_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P040_3_S08_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P041_1_S11_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P041_2_S13_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P041_3_S14_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P042_1_S11_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P042_2_S13_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P042_3_S14_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P043_1_S11_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P043_2_S12_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P043_3_S14_C03.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P044_1_S11_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P044_2_S12_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P044_3_S13_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P045_1_S12_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P045_2_S13_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P045_3_S14_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P046_1_S12_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P046_2_S13_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P046_3_S14_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P047_1_S13_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P047_2_S14_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P047_3_S15_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P048_1_S10_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P048_2_S14_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P048_3_S15_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P049_1_S10_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P049_2_S11_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P049_3_S15_C02.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P050_1_S10_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P050_2_S11_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P050_3_S12_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P051_1_S10_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P051_2_S11_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P051_3_S12_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P052_1_S11_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P052_2_S12_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P052_3_S13_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P053_1_S15_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P053_2_S16_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P053_3_S19_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P054_1_S16_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P054_2_S17_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P054_3_S19_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P055_1_S15_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P055_2_S16_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P055_3_S17_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P056_1_S16_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P056_2_S17_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P056_3_S18_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P057_1_S17_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P057_2_S18_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P057_3_S19_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P058_1_S15_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P058_2_S18_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P058_3_S19_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P059_1_S15_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P059_2_S18_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P059_3_S19_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P060_1_S15_C03.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P060_2_S16_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P060_3_S19_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P061_1_S16_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P061_2_S17_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P061_3_S18_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P062_1_S16_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P062_2_S17_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P062_3_S18_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P063_1_S16_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P063_2_S17_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P063_3_S18_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_eoc\cell_eoc_P064_1_S17_C01.csv"])