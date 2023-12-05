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
uniqueZellen = []

def funktion_lookuptable(x,a0,a1):
    return 1- a0*np.power(x,a1)

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



def cyc_lookuptable(arrayOfPathToCsvFiles):
    socAusSpannung= {2.5000: 0,
                 3.2490: 10, 
                 4.0920: 90,
                 4.2000: 100}
    df_alleZellen = pd.DataFrame(columns=['nameCell', 'xdata', 'ydata','age_chg_rate','age_dischg_rate','age_temp','a0','a1'])
    parameterSatzLaenge = {}

    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Cyclic\cyc_funktion8_expo.csv"
    listOfPrograms = []
    
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {273.15:":",283.15:"--",298.15:"-.", 313.15:"-"}
    # lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cyc_lookup_0_100_0.33_0.33")
    fig2 = plt.figure("cyc_lookup_0_100_0.33_1")
    fig3 = plt.figure("cyc_lookup_0_100_1_1")
    fig4 = plt.figure("cyc_lookup_0_100_1.67_1")
    fig5 = plt.figure("cyc_lookup_10_100_0.33_0.33")
    fig6 = plt.figure("cyc_lookup_10_100_0.33_1")
    fig7 = plt.figure("cyc_lookup_10_100_1_1")
    fig8 = plt.figure("cyc_lookup_10_100_1.67_1")
    fig9 = plt.figure("cyc_lookup_10_90_0.33_0.33")
    fig10= plt.figure("cyc_lookup_10_90_0.33_1")
    fig11= plt.figure("cyc_lookup_10_90_1_1")
    fig12= plt.figure("cyc_lookup_10_90_1.67_1")

    parameterSatzFig = {(0,100,0.33,0.33):fig1 ,
                      (0,100,0.33,1): fig2,
                      (0,100,1,1): fig3,
                      (0,100,1.67,1): fig4,
                      (10,100,0.33,0.33): fig5,
                      (10,100,0.33,1): fig6,
                      (10,100,1,1): fig7,
                      (10,100,1.67,1): fig8,
                      (10,90,0.33,0.33): fig9,
                      (10,90,0.33,1): fig10,
                      (10,90,1,1): fig11,
                      (10,90,1.67,1): fig12}


    
    for pathToCell in arrayOfPathToCsvFiles:
        

        temp_df = pd.read_csv(pathToCell,usecols=["cyc_condition","cyc_charged","v_max_target_V","v_min_target_V","age_temp","age_dischg_rate","age_chg_rate"],sep=";",header =0, nrows=500)
        temp_df = temp_df[(temp_df["cyc_condition"]==1) &(temp_df["cyc_charged"]==1)]
        v_max_target_V = temp_df["v_max_target_V"].iloc[0]
        v_min_target_V = temp_df["v_min_target_V"].iloc[0]
        dsoc = socAusSpannung[v_max_target_V]-socAusSpannung[v_min_target_V]
        midsoc = (socAusSpannung[v_max_target_V]+socAusSpannung[v_min_target_V])/2
        age_temp = temp_df["age_temp"].iloc[0]+273.15
        age_temp = float(age_temp)
        age_dischg_rate = temp_df["age_dischg_rate"].iloc[0]
        age_chg_rate = temp_df["age_chg_rate"].iloc[0]





        df = pd.read_csv(pathToCell, header=0, delimiter=";")
        df = df[(df['cyc_charged'] == 0) & (df['cyc_condition'] == 2)]
        relevant_df = df[['total_q_chg_sum_Ah', 'cap_aged_est_Ah']].copy()        
        relevant_df.loc[:, 'cap_aged_est_Ah'] = relevant_df['cap_aged_est_Ah'] / relevant_df['cap_aged_est_Ah'].iloc[0]
        relevant_df = relevant_df.reset_index()
        xdata = relevant_df['total_q_chg_sum_Ah'].values
        xdata = xdata/(3) #Normierung auf EFC
        


        ydata = relevant_df['cap_aged_est_Ah'].values
        
        parameterSatz = (socAusSpannung[v_min_target_V],socAusSpannung[v_max_target_V],age_chg_rate,age_dischg_rate)
        if parameterSatz not in listOfPrograms:
            print(f"Tuple not found in the list, adding {parameterSatz}")
            listOfPrograms.append(parameterSatz)
        if parameterSatz not in parameterSatzLaenge:
            parameterSatzLaenge[parameterSatz] = xdata[-1]
        elif parameterSatzLaenge[parameterSatz] < xdata[-1]:
            parameterSatzLaenge[parameterSatz] = xdata[-1]

            

        sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_temp], ax = parameterSatzFig[parameterSatz].gca(), alpha = 0.5)

        p0 = [0.2,0.5]
        lower_bounds = [0.001,0.4]
        upper_bounds = [0.2,0.8]
        bounds=(lower_bounds, upper_bounds)
        popt, pcov = optimize.curve_fit(funktion_lookuptable, xdata, ydata, p0=p0, bounds=bounds)
        print("popt", popt)
        

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xdata': [xdata], 'ydata': [ydata], 'dsoc':dsoc, 'age_chg_rate':age_chg_rate, 'age_dischg_rate':age_dischg_rate, 'age_temp':age_temp, 'a0': popt[0], 'a1': popt[1]}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    # df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()


#    (0,100,0.33,0.33)


    tempArray = [273.15, 283.15, 298.15, 313.15]
    errorList = []
    globalErrorList = []
    for parameterSatz in listOfPrograms:
        parameterSatzFig[parameterSatz].gca().set_ylim(0.75, 1.05)
        errorList=[]
        # parameterSatzFig[parameterSatz].gca().set_ylim(0.75, 1.05)
        for temperature in tempArray:
            age_temp = temperature
            dsoc = parameterSatz[1]-parameterSatz[0]
            age_chg_rate=parameterSatz[2]
            age_dischg_rate = parameterSatz[3]

            df_temp = df_alleZellen[(df_alleZellen["dsoc"]==parameterSatz[1]-parameterSatz[0])&(df_alleZellen["age_temp"]==age_temp)&(df_alleZellen["age_chg_rate"]==age_chg_rate)&(df_alleZellen["age_dischg_rate"]==age_dischg_rate)]
            
            popt = df_temp.drop("nameCell", axis=1).drop("xdata", axis=1).drop("ydata", axis=1).drop("age_chg_rate", axis=1).drop("age_dischg_rate", axis=1).drop("age_temp", axis=1).drop("dsoc", axis=1).mean()
            xdata = np.linspace(0, parameterSatzLaenge[parameterSatz], 40)
            
            sns.lineplot(x = xdata, y = funktion_lookuptable(xdata, *popt), color = colorPalette[age_temp], ax = parameterSatzFig[parameterSatz].gca(),linestyle = lineStyle[age_temp])
            # Calculate error
            for cell in uniqueZellen:
                if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_chg_rate"].iloc[0] == age_chg_rate) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_dischg_rate"].iloc[0] == age_dischg_rate)& (df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0] == dsoc)& (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0] == age_temp):
                    xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xdata"].iloc[0]
                    ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["ydata"].iloc[0]
                    ydata = ydata[ydata>0.8]
                    xdata = xdata[:len(ydata)]
                    # age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
                    
                    y_modellierte= funktion_lookuptable(xdata, *popt)
                    error = (y_modellierte-ydata)**2
                    errorList.append(np.mean(error*1000))
                    globalErrorList.append(np.mean(error*1000))
        
        
        
        # print(f"RSME SOC age_voltage: {rmseParameterSatz}")
        rmseParameterSatz = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")
        rmseString = "%.3f" % rmseParameterSatz
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        line4 = plt.Line2D([0], [0], color=colorPalette[273.15], lw=0, label ="Temperatur:")
        line5 = plt.Line2D([0], [0], color=colorPalette[273.15], lw=2,linestyle=':', label ="0°C")
        line6 = plt.Line2D([0], [0], color=colorPalette[283.15], lw=2,linestyle='--', label ="10°C")
        line7 = plt.Line2D([0], [0], color=colorPalette[298.15], lw=2,linestyle='-.', label ="25°C")
        line8 = plt.Line2D([0], [0], color=colorPalette[313.15], lw=2,linestyle='-',label ="40°C")
        
        legend1 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4,line5, line6,line7,line8], loc='upper right')

       

    
        parameterSatzFig[parameterSatz].gca().add_artist(legend1)



        parameterSatzFig[parameterSatz].gca().grid(color='lightgrey', linestyle='--')
        parameterSatzFig[parameterSatz].gca().set_xlabel(f'EFC (-)')
        parameterSatzFig[parameterSatz].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        parameterSatzFig[parameterSatz].gca().set_title(f'Zyklisch LookUp - {parameterSatz[0]}-{parameterSatz[1]}%  {parameterSatz[2]}C {parameterSatz[3]}C')




    rsme = "%.4f" % np.sqrt(np.mean(globalErrorList))
    print(f"Root Mean Squared Error: {rsme}")
    # optimal_parameters = result.x
    # optimal_function_value = result.fun

    # csv_filename = fileToSaveValues
    # header = ['Parameter {}'.format(i) for i in range(len(optimal_parameters))] + ['Optimal Function Value'] +['RSME']
    # data = [np.array(optimal_parameters).tolist() + [optimal_function_value] + [np.sqrt(np.mean(globalErrorList))]]

    # # Write the data to the CSV file
    # with open(csv_filename, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile,delimiter =';')
    #     csv_writer.writerow(header)
    #     csv_writer.writerows(data)

    
    
    saveFigsDirectory = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Cyclic\LookUpTable\\"
    fig1.savefig(saveFigsDirectory+"cyc_lookup_0_100_0.33_0.33.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cyc_lookup_0_100_0.33_1.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cyc_lookup_0_100_1_1.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cyc_lookup_0_100_1.67_1.png", format='png',  pad_inches=0, transparent=False)
    fig5.savefig(saveFigsDirectory+"cyc_lookup_10_100_0.33_0.33.png", format='png',  pad_inches=0, transparent=False)
    fig6.savefig(saveFigsDirectory+"cyc_lookup_10_100_0.33_1.png", format='png',  pad_inches=0, transparent=False)
    fig7.savefig(saveFigsDirectory+"cyc_lookup_10_100_1_1.png", format='png',  pad_inches=0, transparent=False)
    fig8.savefig(saveFigsDirectory+"cyc_lookup_10_100_1.67_1.png", format='png',  pad_inches=0, transparent=False)
    fig9.savefig(saveFigsDirectory+"cyc_lookup_10_90_0.33_0.33.png", format='png',  pad_inches=0, transparent=False)
    fig10.savefig(saveFigsDirectory+"cyc_lookup_10_90_0.33_1.png", format='png',  pad_inches=0, transparent=False)
    fig11.savefig(saveFigsDirectory+"cyc_lookup_10_90_1_1.png", format='png',  pad_inches=0, transparent=False)
    fig12.savefig(saveFigsDirectory+"cyc_lookup_10_90_1.67_1.png", format='png',  pad_inches=0, transparent=False)


    plt.show()





        
   

    
















if __name__ == '__main__':



    cyc_lookuptable([r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P062_3_S18_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P063_1_S16_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P063_2_S17_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P063_3_S18_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P064_1_S17_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P064_2_S18_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P064_3_S19_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P017_1_S01_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P017_2_S04_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P017_3_S05_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P018_1_S01_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P018_2_S02_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P018_3_S04_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P019_1_S01_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P019_2_S02_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P019_3_S05_C02.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P020_1_S01_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P020_2_S02_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P020_3_S03_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P021_1_S02_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P021_2_S03_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P021_3_S04_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P022_1_S03_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P022_2_S04_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P022_3_S05_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P023_1_S03_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P023_2_S04_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P023_3_S05_C03.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P024_1_S01_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P024_2_S04_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P024_3_S05_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P025_1_S01_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P025_2_S02_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P025_3_S03_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P026_1_S01_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P026_2_S02_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P026_3_S03_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P027_1_S01_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P027_2_S02_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P027_3_S03_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P028_1_S02_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P028_2_S03_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P028_3_S04_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P029_1_S06_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P029_2_S08_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P029_3_S09_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P030_1_S06_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P030_2_S08_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P030_3_S09_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P031_1_S06_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P031_2_S07_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P031_3_S09_C03.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P032_1_S06_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P032_2_S07_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P032_3_S08_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P033_1_S07_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P033_2_S08_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P033_3_S09_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P034_1_S07_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P034_2_S08_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P034_3_S09_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P035_1_S08_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P035_2_S09_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P035_3_S10_C02.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P036_1_S05_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P036_2_S09_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P036_3_S10_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P037_1_S05_C11.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P037_2_S06_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P037_3_S10_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P038_1_S06_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P038_2_S07_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P038_3_S10_C03.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P039_1_S05_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P039_2_S06_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P039_3_S07_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P040_1_S06_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P040_2_S07_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P040_3_S08_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P041_1_S11_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P041_2_S13_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P041_3_S14_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P042_1_S11_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P042_2_S13_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P042_3_S14_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P043_1_S11_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P043_2_S12_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P043_3_S14_C03.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P044_1_S11_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P044_2_S12_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P044_3_S13_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P045_1_S12_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P045_2_S13_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P045_3_S14_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P046_1_S12_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P046_2_S13_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P046_3_S14_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P047_1_S13_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P047_2_S14_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P047_3_S15_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P048_1_S10_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P048_2_S14_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P048_3_S15_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P049_1_S10_C10.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P049_2_S11_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P049_3_S15_C02.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P050_1_S10_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P050_2_S11_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P050_3_S12_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P051_1_S10_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P051_2_S11_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P051_3_S12_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P052_1_S11_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P052_2_S12_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P052_3_S13_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P053_1_S15_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P053_2_S16_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P053_3_S19_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P054_1_S16_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P054_2_S17_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P054_3_S19_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P055_1_S15_C05.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P055_2_S16_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P055_3_S17_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P056_1_S16_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P056_2_S17_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P056_3_S18_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P057_1_S17_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P057_2_S18_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P057_3_S19_C08.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P058_1_S15_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P058_2_S18_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P058_3_S19_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P059_1_S15_C06.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P059_2_S18_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P059_3_S19_C04.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P060_1_S15_C03.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P060_2_S16_C01.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P060_3_S19_C00.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P061_1_S16_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P061_2_S17_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P061_3_S18_C09.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P062_1_S16_C07.csv",r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P062_2_S17_C07.csv"])