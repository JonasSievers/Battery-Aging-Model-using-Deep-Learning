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
def funktionExponentMitWerten(x):
    return 0.9272072696667285-0.0072208866623697365*x+4.6916412031700374e-05*x*x
def funktion_lookuptable(x,a0):
    return 1- a0*np.power(x,exponent)

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
