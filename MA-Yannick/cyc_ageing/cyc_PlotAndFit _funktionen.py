

    
import csv
import re
import time
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
import seaborn as sns

tempFilesDirectory = "C:\\Users\\Yannick\\Desktop\\tempSkripte\\"
saveFigsDirectory = "C:\\Users\\Yannick\\bwSyncShare\\MA Yannick Fritsch\\02_Arbeit\\Grafiken\\"

dsoc = 1
midsoc= 0
age_chg_rate = 0
age_dischg_rate= 0
age_temp= 0

socAusSpannung= {2.5000: 0,
                 3.2490: 10, 
                 4.0920: 90,
                 4.2000: 100}
df_alleZellen = pd.DataFrame(columns=['nameCell', 'xdata', 'ydata','midsoc', 'dsoc', 'age_chg_rate', 'age_dischg_rate', 'age_temp'])

def funktion5(x, a0,a1):
    return 1 - (a0*np.exp(a1*dsoc/age_temp)*np.power(x,0.5)) #    (a0*np.exp(a1*dsoc) + a2*np.exp(a3*dsoc/age_temp)*np.power(x,0.5))
def funktion5_expo(x, a0, a1, a2):
    return 1 - (a0*np.exp(a1*dsoc/age_temp)*np.power(x,a2))
def funktion5_min_expo(parameters):
    global uniqueZellen,df_alleZellen,age_voltage,age_temp,dsoc
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        
        dsoc = df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0]
        
        y_modellierte= funktion5_expo(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors
def funktion5_minimize_expo(arrayOfPathToCsvFiles):
    global age_temp, age_voltage,maxXValue,exponent,uniqueZellen,df_alleZellen
    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Cyclic\cyc_funktion5_expo.csv"
    listOfPrograms = []
    
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {273.15:":",283.15:"--",298.15:"-.", 313.15:"-"}
    # lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cyc_funktion5_0_100_0.33_0.33")
    fig2 = plt.figure("cyc_funktion5_0_100_0.33_1")
    fig3 = plt.figure("cyc_funktion5_0_100_1_1")
    fig4 = plt.figure("cyc_funktion5_0_100_1.67_1")
    fig5 = plt.figure("cyc_funktion5_10_100_0.33_0.33")
    fig6 = plt.figure("cyc_funktion5_10_100_0.33_1")
    fig7 = plt.figure("cyc_funktion5_10_100_1_1")
    fig8 = plt.figure("cyc_funktion5_10_100_1.67_1")
    fig9 = plt.figure("cyc_funktion5_10_90_0.33_0.33")
    fig10= plt.figure("cyc_funktion5_10_90_0.33_1")
    fig11= plt.figure("cyc_funktion5_10_90_1_1")
    fig12= plt.figure("cyc_funktion5_10_90_1.67_1")

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
    parameterSatzMaxXLength = {(0,100,0.33,0.33):0,
                    (0,100,0.33,1):0,
                    (0,100,1,1):0,
                    (0,100,1.67,1):0,
                    (10,100,0.33,0.33):0,
                    (10,100,0.33,1):0,
                    (10,100,1,1):0,
                    (10,100,1.67,1):0,
                    (10,90,0.33,0.33):0,
                    (10,90,0.33,1): 0,
                    (10,90,1,1): 0,
                    (10,90,1.67,1): 0}


    
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
        if max(xdata) > parameterSatzMaxXLength[parameterSatz]:
            parameterSatzMaxXLength[parameterSatz] = max(xdata)

        sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_temp], ax = parameterSatzFig[parameterSatz].gca(), alpha = 0.5)
        

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'midsoc':midsoc, 'dsoc':dsoc, 'age_chg_rate':age_chg_rate, 'age_dischg_rate':age_dischg_rate, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()
    start = [0.05,10,0.6]
    bounds = [(0.001,0.5),(-5,10),(0.1,0.9)]

    startTime = time.time()
    print(f"Starting optimization {startTime}")
    result = optimize.minimize(funktion5_min_expo,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [-3.000e-01, -1.000e+00,  1.911e-03,  2.863e-02,  3.218e-01]
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    tempArray = [273.15, 283.15, 298.15, 313.15]
    errorList = []
    globalErrorList = []
    for parameterSatz in listOfPrograms:
        errorList=[]
        parameterSatzFig[parameterSatz].gca().set_ylim(0.75, 1.05)
        for temperature in tempArray:
            age_temp = temperature
            dsoc = parameterSatz[1]-parameterSatz[0]
            age_chg_rate = parameterSatz[2]
            age_dischg_rate =parameterSatz[3]
            xdata_sim = np.linspace(0,parameterSatzMaxXLength[parameterSatz],20)
            sns.lineplot(x = xdata_sim, y = funktion5_expo(xdata_sim, *result.x), color = colorPalette[age_temp], ax = parameterSatzFig[parameterSatz].gca(),linestyle = lineStyle[age_temp])

        # Calculate error
        for cell in uniqueZellen:
            if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_chg_rate"].iloc[0] == age_chg_rate) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_dischg_rate"].iloc[0] == age_dischg_rate)& (df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0] == dsoc):
                xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                ydata = ydata[ydata>0.8]
                xdata = xdata[:len(ydata)]
                # age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
                
                y_modellierte= funktion5_expo(xdata, *result.x)
                error = (y_modellierte-ydata)**2
                errorList.append(np.mean(error*1000))
                globalErrorList.append(np.mean(error*1000))
        # print(f"RSME SOC age_voltage: {rmseParameterSatz}")
        rmseParameterSatz = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=0, label ="SOC")
        line2 = plt.Line2D([0], [0], color="grey", lw=2,linestyle=':', label ="10%")
        line3 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='--', label ="50%")
        line4 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-.', label ="90%")
        line5 = plt.Line2D([0], [0], color="grey", lw=2,linestyle='-',label ="100%")
        legend1 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = "%.3f" % result.x[-1]
        rmseString = "%.3f" % rmseParameterSatz
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        parameterSatzFig[parameterSatz].gca().add_artist(legend1)
        parameterSatzFig[parameterSatz].gca().add_artist(legend2)


        parameterSatzFig[parameterSatz].gca().grid(color='lightgrey', linestyle='--')
        parameterSatzFig[parameterSatz].gca().set_xlabel(f'EFC (-)')
        parameterSatzFig[parameterSatz].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        parameterSatzFig[parameterSatz].gca().set_title(f'Zyklische Modellierung - Funktion Nr. 5 - {parameterSatz[0]}-{parameterSatz[1]}%  {parameterSatz[2]}C {parameterSatz[3]}C')




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


    fig1.savefig(saveFigsDirectory+"cyc_funktion5_0_100_0.33_0.33.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cyc_funktion5_0_100_0.33_1.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cyc_funktion5_0_100_1_1.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cyc_funktion5_0_100_1.67_1.png", format='png',  pad_inches=0, transparent=False)
    fig5.savefig(saveFigsDirectory+"cyc_funktion5_10_100_0.33_0.33.png", format='png',  pad_inches=0, transparent=False)
    fig6.savefig(saveFigsDirectory+"cyc_funktion5_10_100_0.33_1.png", format='png',  pad_inches=0, transparent=False)
    fig7.savefig(saveFigsDirectory+"cyc_funktion5_10_100_1_1.png", format='png',  pad_inches=0, transparent=False)
    fig8.savefig(saveFigsDirectory+"cyc_funktion5_10_100_1.67_1.png", format='png',  pad_inches=0, transparent=False)
    fig9.savefig(saveFigsDirectory+"cyc_funktion5_10_90_0.33_0.33.png", format='png',  pad_inches=0, transparent=False)
    fig10.savefig(saveFigsDirectory+"cyc_funktion5_10_90_0.33_1.png", format='png',  pad_inches=0, transparent=False)
    fig11.savefig(saveFigsDirectory+"cyc_funktion5_10_90_1_1.png", format='png',  pad_inches=0, transparent=False)
    fig12.savefig(saveFigsDirectory+"cyc_funktion5_10_90_1.67_1.png", format='png',  pad_inches=0, transparent=False)


    plt.show()

def funktion5_min(parameters):
    global uniqueZellen,df_alleZellen,age_voltage,age_temp,dsoc
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        
        dsoc = df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0]
        
        y_modellierte= funktion5(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors
def funktion5_minimize(arrayOfPathToCsvFiles):
    global age_temp, age_voltage,maxXValue,exponent,uniqueZellen,df_alleZellen
    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Cyclic\cyc_funktion5_fixed.csv"
    listOfPrograms = []
    
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {273.15:":",283.15:"--",298.15:"-.", 313.15:"-"}
    # lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cyc_funktion5_0_100_0.33_0.33")
    fig2 = plt.figure("cyc_funktion5_0_100_0.33_1")
    fig3 = plt.figure("cyc_funktion5_0_100_1_1")
    fig4 = plt.figure("cyc_funktion5_0_100_1.67_1")
    fig5 = plt.figure("cyc_funktion5_10_100_0.33_0.33")
    fig6 = plt.figure("cyc_funktion5_10_100_0.33_1")
    fig7 = plt.figure("cyc_funktion5_10_100_1_1")
    fig8 = plt.figure("cyc_funktion5_10_100_1.67_1")
    fig9 = plt.figure("cyc_funktion5_10_90_0.33_0.33")
    fig10= plt.figure("cyc_funktion5_10_90_0.33_1")
    fig11= plt.figure("cyc_funktion5_10_90_1_1")
    fig12= plt.figure("cyc_funktion5_10_90_1.67_1")

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

        sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_temp], ax = parameterSatzFig[parameterSatz].gca(), alpha = 0.5)
        

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'midsoc':midsoc, 'dsoc':dsoc, 'age_chg_rate':age_chg_rate, 'age_dischg_rate':age_dischg_rate, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()

    start = [0.05,10]
    bounds = [(0.001,0.5),(-5,10)]
    startTime = time.time()
    print(f"Starting optimization {startTime}")
    result = optimize.minimize(funktion5_min,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [-3.000e-01, -1.000e+00,  1.911e-03,  2.863e-02,  3.218e-01]
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    tempArray = [273.15, 283.15, 298.15, 313.15]
    errorList = []
    globalErrorList = []
    for parameterSatz in listOfPrograms:
        errorList=[]
        parameterSatzFig[parameterSatz].gca().set_ylim(0.75, 1.05)
        for temperature in tempArray:
            age_temp = temperature
            dsoc = parameterSatz[1]-parameterSatz[0]
            age_chg_rate = parameterSatz[2]
            age_dischg_rate =parameterSatz[3]
            xdata = np.linspace(0,1200,20)

            sns.lineplot(x = xdata, y = funktion5(xdata, *result.x), color = colorPalette[age_temp], ax = parameterSatzFig[parameterSatz].gca(),linestyle = lineStyle[age_temp])

        # Calculate error
        for cell in uniqueZellen:
            if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_chg_rate"].iloc[0] == age_chg_rate) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_dischg_rate"].iloc[0] == age_dischg_rate)& (df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0] == dsoc):
                xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                ydata = ydata[ydata>0.8]
                xdata = xdata[:len(ydata)]
                # age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
                
                y_modellierte= funktion5(xdata, *result.x)
                error = (y_modellierte-ydata)**2
                errorList.append(np.mean(error*1000))
                globalErrorList.append(np.mean(error*1000))
        # print(f"RSME SOC age_voltage: {rmseParameterSatz}")
        rmseParameterSatz = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[273.15], lw=0, label ="Temperatur")
        line2 = plt.Line2D([0], [0], color=colorPalette[273.15], lw=2,linestyle=':', label ="0°C")
        line3 = plt.Line2D([0], [0], color=colorPalette[283.15], lw=2,linestyle='--', label ="10°C")
        line4 = plt.Line2D([0], [0], color=colorPalette[298.15], lw=2,linestyle='-.', label ="25°C")
        line5 = plt.Line2D([0], [0], color=colorPalette[313.15], lw=2,linestyle='-',label ="40°C")
        legend1 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = 0.5 #"%.3f" % result.x[4]
        rmseString = "%.3f" % rmseParameterSatz
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        parameterSatzFig[parameterSatz].gca().add_artist(legend1)
        parameterSatzFig[parameterSatz].gca().add_artist(legend2)


        parameterSatzFig[parameterSatz].gca().grid(color='lightgrey', linestyle='--')
        parameterSatzFig[parameterSatz].gca().set_xlabel(f'EFC (-)')
        parameterSatzFig[parameterSatz].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        parameterSatzFig[parameterSatz].gca().set_title(f'Zyklische Modellierung - Funktion Nr. 5 - {parameterSatz[0]}-{parameterSatz[1]}%  {parameterSatz[2]}C {parameterSatz[3]}C')




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


    fig1.savefig(saveFigsDirectory+"cyc_funktion5_0_100_0.33_0.33_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cyc_funktion5_0_100_1.67_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cyc_funktion5_0_100_0.33_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cyc_funktion5_0_100_1_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig5.savefig(saveFigsDirectory+"cyc_funktion5_10_100_0.33_0.33_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig6.savefig(saveFigsDirectory+"cyc_funktion5_10_100_0.33_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig7.savefig(saveFigsDirectory+"cyc_funktion5_10_100_1_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig8.savefig(saveFigsDirectory+"cyc_funktion5_10_100_1.67_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig9.savefig(saveFigsDirectory+"cyc_funktion5_10_90_0.33_0.33_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig10.savefig(saveFigsDirectory+"cyc_funktion5_10_90_0.33_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig11.savefig(saveFigsDirectory+"cyc_funktion5_10_90_1_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig12.savefig(saveFigsDirectory+"cyc_funktion5_10_90_1.67_1_fixed.png", format='png',  pad_inches=0, transparent=False)


    plt.show()


def funktion7(x, a0, a1, a2):
    return 1- (a0*(1+a1*(midsoc-50)*(midsoc-50))*dsoc/100*age_chg_rate/1000*np.exp(a2/age_temp))*np.power(x,0.5)
    return 1 - (a0*(1+a1*(midsoc-50)*(midsoc-50)) *dsoc*(age_chg_rate*age_chg_rate)*np.exp(age_temp/a2)*np.power(x,0.5))
    return 1 - (a0*dsoc*(age_chg_rate*age_chg_rate+a1*age_dischg_rate)*np.exp(a2/age_temp))*np.power(x,0.5)
def funktion7_expo(x, a0, a1, a2,a3):
    return 1- (a0*(1+a1*(midsoc-50)*(midsoc-50))*dsoc/100*age_chg_rate/1000*np.exp(a2/age_temp))*np.power(x,a3)
    return 1 - (a0*(1+a1*np.power(midsoc-0.5,a2))*dsoc*(age_chg_rate*age_chg_rate+a3*age_dischg_rate)*np.exp(a4/age_temp))*np.power(x,a5)
def funktion7_min(parameters):
    global uniqueZellen,df_alleZellen,age_voltage,age_temp,dsoc,midsoc,age_chg_rate,age_dischg_rate
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        midsoc =df_alleZellen[df_alleZellen["nameCell"]==cell]["midsoc"].iloc[0]
        dsoc = df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0]
        age_chg_rate = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_chg_rate"].iloc[0]
        age_dischg_rate = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_dischg_rate"].iloc[0]
        age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
        y_modellierte= funktion7(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors
def funktion7_minimize(arrayOfPathToCsvFiles):
    global age_temp, age_voltage,maxXValue,exponent,uniqueZellen,df_alleZellen,midsoc,dsoc
    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Cyclic\cyc_funktion7_fixed.csv"
    listOfPrograms = []
    
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {273.15:":",283.15:"--",298.15:"-.", 313.15:"-"}
    # lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cyc_funktion7_0_100_0.33_0.33")
    fig2 = plt.figure("cyc_funktion7_0_100_0.33_1")
    fig3 = plt.figure("cyc_funktion7_0_100_1_1")
    fig4 = plt.figure("cyc_funktion7_0_100_1.67_1")
    fig5 = plt.figure("cyc_funktion7_10_100_0.33_0.33")
    fig6 = plt.figure("cyc_funktion7_10_100_0.33_1")
    fig7 = plt.figure("cyc_funktion7_10_100_1_1")
    fig8 = plt.figure("cyc_funktion7_10_100_1.67_1")
    fig9 = plt.figure("cyc_funktion7_10_90_0.33_0.33")
    fig10= plt.figure("cyc_funktion7_10_90_0.33_1")
    fig11= plt.figure("cyc_funktion7_10_90_1_1")
    fig12= plt.figure("cyc_funktion7_10_90_1.67_1")

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

        sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_temp], ax = parameterSatzFig[parameterSatz].gca(), alpha = 0.5)
        

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'midsoc':midsoc, 'dsoc':dsoc, 'age_chg_rate':age_chg_rate, 'age_dischg_rate':age_dischg_rate, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()

    start = [0.1,0.1,1000]
    bounds = [(0,1),(0,1),(100,500)]
    startTime = time.time()
    print(f"Starting optimization {startTime}")
    result = optimize.minimize(funktion7_min,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [ 2.327e-05, -7.495e-01, -1.178e+01]
    #         self.fun = 0
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    tempArray = [273.15, 283.15, 298.15, 313.15]
    errorList = []
    globalErrorList = []
    for parameterSatz in listOfPrograms:
        errorList=[]
        parameterSatzFig[parameterSatz].gca().set_ylim(0.75, 1.05)
        for temperature in tempArray:
            age_temp = temperature
            dsoc = parameterSatz[1]-parameterSatz[0]
            midsoc = (parameterSatz[1]+parameterSatz[0])/2
            age_chg_rate = parameterSatz[2]
            age_dischg_rate =parameterSatz[3]
            xdata = np.linspace(0,1400,12)
            sns.lineplot(x = xdata, y = funktion7(xdata, *result.x), color = colorPalette[age_temp], ax = parameterSatzFig[parameterSatz].gca(),linestyle = lineStyle[age_temp])

        # Calculate error
        for cell in uniqueZellen:
            if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_chg_rate"].iloc[0] == age_chg_rate) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_dischg_rate"].iloc[0] == age_dischg_rate)& (df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0] == dsoc):
                xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                ydata = ydata[ydata>0.8]
                xdata = xdata[:len(ydata)]
                
                
                
                
                y_modellierte= funktion7(xdata, *result.x)
                error = (y_modellierte-ydata)**2
                errorList.append(np.mean(error*1000))
                globalErrorList.append(np.mean(error*1000))
        # print(f"RSME SOC age_voltage: {rmseParameterSatz}")
        rmseParameterSatz = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=0, label ="Temperatur")
        line2 = plt.Line2D([0], [0], color=colorPalette[273.15], lw=2,linestyle=':', label ="0°C")
        line3 = plt.Line2D([0], [0], color=colorPalette[283.15], lw=2,linestyle='--', label ="10°C")
        line4 = plt.Line2D([0], [0], color=colorPalette[298.15], lw=2,linestyle='-.', label ="25°C")
        line5 = plt.Line2D([0], [0], color=colorPalette[313.15], lw=2,linestyle='-',label ="40°C")
        legend1 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = 0.5 #"%.3f" % result.x[4]
        rmseString = "%.3f" % rmseParameterSatz
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        parameterSatzFig[parameterSatz].gca().add_artist(legend1)
        parameterSatzFig[parameterSatz].gca().add_artist(legend2)


        parameterSatzFig[parameterSatz].gca().grid(color='lightgrey', linestyle='--')
        parameterSatzFig[parameterSatz].gca().set_xlabel(f'EFC (-)')
        parameterSatzFig[parameterSatz].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        parameterSatzFig[parameterSatz].gca().set_title(f'Zyklische Modellierung - Funktion Nr. 7 - {parameterSatz[0]}-{parameterSatz[1]}%  {parameterSatz[2]}C {parameterSatz[3]}C')




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


    fig1.savefig(saveFigsDirectory+"cyc_funktion7_0_100_0.33_0.33_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cyc_funktion7_0_100_0.33_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cyc_funktion7_0_100_1_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cyc_funktion7_0_100_1.67_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig5.savefig(saveFigsDirectory+"cyc_funktion7_10_100_0.33_0.33_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig6.savefig(saveFigsDirectory+"cyc_funktion7_10_100_0.33_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig7.savefig(saveFigsDirectory+"cyc_funktion7_10_100_1_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig8.savefig(saveFigsDirectory+"cyc_funktion7_10_100_1.67_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig9.savefig(saveFigsDirectory+"cyc_funktion7_10_90_0.33_0.33_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig10.savefig(saveFigsDirectory+"cyc_funktion7_10_90_0.33_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig11.savefig(saveFigsDirectory+"cyc_funktion7_10_90_1_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig12.savefig(saveFigsDirectory+"cyc_funktion7_10_90_1.67_1_fixed.png", format='png',  pad_inches=0, transparent=False)


    plt.show()
def funktion7_min_expo(parameters):
    global uniqueZellen,df_alleZellen,age_voltage,age_temp,dsoc,midsoc,age_chg_rate,age_dischg_rate
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        midsoc =df_alleZellen[df_alleZellen["nameCell"]==cell]["midsoc"].iloc[0]
        dsoc = df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0]
        age_chg_rate = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_chg_rate"].iloc[0]
        age_dischg_rate = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_dischg_rate"].iloc[0]
        age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
        y_modellierte= funktion7_expo(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors
def funktion7_minimize_expo(arrayOfPathToCsvFiles):
    global age_temp, age_voltage,maxXValue,exponent,uniqueZellen,df_alleZellen,midsoc,dsoc
    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Cyclic\cyc_funktion7_expo.csv"
    listOfPrograms = []
    
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {273.15:":",283.15:"--",298.15:"-.", 313.15:"-"}
    # lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cyc_funktion7_0_100_0.33_0.33")
    fig2 = plt.figure("cyc_funktion7_0_100_0.33_1")
    fig3 = plt.figure("cyc_funktion7_0_100_1_1")
    fig4 = plt.figure("cyc_funktion7_0_100_1.67_1")
    fig5 = plt.figure("cyc_funktion7_10_100_0.33_0.33")
    fig6 = plt.figure("cyc_funktion7_10_100_0.33_1")
    fig7 = plt.figure("cyc_funktion7_10_100_1_1")
    fig8 = plt.figure("cyc_funktion7_10_100_1.67_1")
    fig9 = plt.figure("cyc_funktion7_10_90_0.33_0.33")
    fig10= plt.figure("cyc_funktion7_10_90_0.33_1")
    fig11= plt.figure("cyc_funktion7_10_90_1_1")
    fig12= plt.figure("cyc_funktion7_10_90_1.67_1")

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

        sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_temp], ax = parameterSatzFig[parameterSatz].gca(), alpha = 0.5)
        

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'midsoc':midsoc, 'dsoc':dsoc, 'age_chg_rate':age_chg_rate, 'age_dischg_rate':age_dischg_rate, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()

    start = [0.01,0.001,1000,0.6]
    bounds = [(0,0.2),(0,0.1),(100,1500),(0.401,0.8)]
    startTime = time.time()
    print(f"Starting optimization {startTime}")
    result = optimize.minimize(funktion7_min_expo,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [ 2.327e-05, -7.495e-01, -1.178e+01]
    #         self.fun = 0
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    tempArray = [273.15, 283.15, 298.15, 313.15]
    errorList = []
    globalErrorList = []
    for parameterSatz in listOfPrograms:
        errorList=[]
        parameterSatzFig[parameterSatz].gca().set_ylim(0.75, 1.05)
        for temperature in tempArray:
            age_temp = temperature
            dsoc = parameterSatz[1]-parameterSatz[0]
            midsoc = (parameterSatz[1]+parameterSatz[0])/2
            age_chg_rate = parameterSatz[2]
            age_dischg_rate =parameterSatz[3]
            xdata = np.linspace(0,1400,12)
            sns.lineplot(x = xdata, y = funktion7_expo(xdata, *result.x), color = colorPalette[age_temp], ax = parameterSatzFig[parameterSatz].gca(),linestyle = lineStyle[age_temp])

        # Calculate error
        for cell in uniqueZellen:
            if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_chg_rate"].iloc[0] == age_chg_rate) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_dischg_rate"].iloc[0] == age_dischg_rate)& (df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0] == dsoc):
                xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                ydata = ydata[ydata>0.8]
                xdata = xdata[:len(ydata)]
                
                
                
                
                y_modellierte= funktion7_expo(xdata, *result.x)
                error = (y_modellierte-ydata)**2
                errorList.append(np.mean(error*1000))
                globalErrorList.append(np.mean(error*1000))
        # print(f"RSME SOC age_voltage: {rmseParameterSatz}")
        rmseParameterSatz = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=0, label ="Temperatur")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle=':', label ="0°C")
        line3 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='--', label ="10°C")
        line4 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-.', label ="25°C")
        line5 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2,linestyle='-',label ="40°C")
        legend1 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4,line5], loc='right')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = "%.3f" % result.x[-1]
        rmseString = "%.3f" % rmseParameterSatz
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        parameterSatzFig[parameterSatz].gca().add_artist(legend1)
        parameterSatzFig[parameterSatz].gca().add_artist(legend2)


        parameterSatzFig[parameterSatz].gca().grid(color='lightgrey', linestyle='--')
        parameterSatzFig[parameterSatz].gca().set_xlabel(f'EFC (-)')
        parameterSatzFig[parameterSatz].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        parameterSatzFig[parameterSatz].gca().set_title(f'Zyklische Modellierung - Funktion Nr. 7 - {parameterSatz[0]}-{parameterSatz[1]}%  {parameterSatz[2]}C {parameterSatz[3]}C')




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


    fig1.savefig(saveFigsDirectory+"cyc_funktion7_0_100_0.33_0.33_expo.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cyc_funktion7_0_100_0.33_1_expo.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cyc_funktion7_0_100_1_1_expo.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cyc_funktion7_0_100_1.67_1_expo.png", format='png',  pad_inches=0, transparent=False)
    fig5.savefig(saveFigsDirectory+"cyc_funktion7_10_100_0.33_0.33_expo.png", format='png',  pad_inches=0, transparent=False)
    fig6.savefig(saveFigsDirectory+"cyc_funktion7_10_100_0.33_1_expo.png", format='png',  pad_inches=0, transparent=False)
    fig7.savefig(saveFigsDirectory+"cyc_funktion7_10_100_1_1_expo.png", format='png',  pad_inches=0, transparent=False)
    fig8.savefig(saveFigsDirectory+"cyc_funktion7_10_100_1.67_1_expo.png", format='png',  pad_inches=0, transparent=False)
    fig9.savefig(saveFigsDirectory+"cyc_funktion7_10_90_0.33_0.33_expo.png", format='png',  pad_inches=0, transparent=False)
    fig10.savefig(saveFigsDirectory+"cyc_funktion7_10_90_0.33_1_expo.png", format='png',  pad_inches=0, transparent=False)
    fig11.savefig(saveFigsDirectory+"cyc_funktion7_10_90_1_1_expo.png", format='png',  pad_inches=0, transparent=False)
    fig12.savefig(saveFigsDirectory+"cyc_funktion7_10_90_1.67_1_expo.png", format='png',  pad_inches=0, transparent=False)


    plt.show()



def funktion6(x,a0,a1,a2):
    return 1-a0*np.exp(a1*dsoc)*np.exp((a2+100*age_chg_rate)/age_temp)*np.power(x,0.5)

def funktion6_min(parameters):
    global uniqueZellen,df_alleZellen,age_temp,dsoc
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        dsoc = df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0]
        age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
        
        y_modellierte= funktion6(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors

def funktion6_minimize(arrayOfPathToCsvFiles):
    global age_temp,uniqueZellen,df_alleZellen
    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Cyclic\cyc_funktion6_fixed.csv"
    listOfPrograms = []
    
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {273.15:":",283.15:"--",298.15:"-.", 313.15:"-"}
    # lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cyc_funktion6_0_100_0.33_0.33")
    fig2 = plt.figure("cyc_funktion6_0_100_0.33_1")
    fig3 = plt.figure("cyc_funktion6_0_100_1_1")
    fig4 = plt.figure("cyc_funktion6_0_100_1.67_1")
    fig5 = plt.figure("cyc_funktion6_10_100_0.33_0.33")
    fig6 = plt.figure("cyc_funktion6_10_100_0.33_1")
    fig7 = plt.figure("cyc_funktion6_10_100_1_1")
    fig8 = plt.figure("cyc_funktion6_10_100_1.67_1")
    fig9 = plt.figure("cyc_funktion6_10_90_0.33_0.33")
    fig10= plt.figure("cyc_funktion6_10_90_0.33_1")
    fig11= plt.figure("cyc_funktion6_10_90_1_1")
    fig12= plt.figure("cyc_funktion6_10_90_1.67_1")

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

        sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_temp], ax = parameterSatzFig[parameterSatz].gca(), alpha = 0.5)
        

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'midsoc':midsoc, 'dsoc':dsoc, 'age_chg_rate':age_chg_rate, 'age_dischg_rate':age_dischg_rate, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()

    start = [0.00001,0.05,1200]
    bounds = [(0.00001,0.00002),(0.04,0.06),(1000,1300)]
    startTime = time.time()
    print(f"Starting optimization {startTime}")
    result = optimize.minimize(funktion6_min,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [-3.000e-01, -1.000e+00,  1.911e-03,  2.863e-02,  3.218e-01]
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    tempArray = [273.15, 283.15, 298.15, 313.15]
    errorList = []
    globalErrorList = []
    for parameterSatz in listOfPrograms:
        errorList=[]
        parameterSatzFig[parameterSatz].gca().set_ylim(0.75, 1.05)
        for temperature in tempArray:
            age_temp = temperature
            dsoc = parameterSatz[1]-parameterSatz[0]
            age_chg_rate = parameterSatz[2]
            age_dischg_rate =parameterSatz[3]
            xdata = np.linspace(0,1200,12)
            sns.lineplot(x = xdata, y = funktion6(xdata, *result.x), color = colorPalette[age_temp], ax = parameterSatzFig[parameterSatz].gca(),linestyle = lineStyle[age_temp])

        # Calculate error
        for cell in uniqueZellen:
            if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_chg_rate"].iloc[0] == age_chg_rate) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_dischg_rate"].iloc[0] == age_dischg_rate)& (df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0] == dsoc):
                xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                ydata = ydata[ydata>0.8]
                xdata = xdata[:len(ydata)]
                # age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
                
                y_modellierte= funktion6(xdata, *result.x)
                error = (y_modellierte-ydata)**2
                errorList.append(np.mean(error*1000))
                globalErrorList.append(np.mean(error*1000))
        # print(f"RSME SOC age_voltage: {rmseParameterSatz}")
        rmseParameterSatz = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[273.15], lw=0, label ="Temperatur")
        line2 = plt.Line2D([0], [0], color=colorPalette[273.15], lw=2,linestyle=':', label ="0°C")
        line3 = plt.Line2D([0], [0], color=colorPalette[283.15], lw=2,linestyle='--', label ="10°C")
        line4 = plt.Line2D([0], [0], color=colorPalette[298.15], lw=2,linestyle='-.', label ="25°C")
        line5 = plt.Line2D([0], [0], color=colorPalette[313.15], lw=2,linestyle='-',label ="40°C")
        legend1 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = 0.5 #"%.3f" % result.x[4]
        rmseString = "%.3f" % rmseParameterSatz
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        parameterSatzFig[parameterSatz].gca().add_artist(legend1)
        parameterSatzFig[parameterSatz].gca().add_artist(legend2)


        parameterSatzFig[parameterSatz].gca().grid(color='lightgrey', linestyle='--')
        parameterSatzFig[parameterSatz].gca().set_xlabel(f'EFC (-)')
        parameterSatzFig[parameterSatz].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        parameterSatzFig[parameterSatz].gca().set_title(f'Zyklische Modellierung - Funktion Nr. 6 - {parameterSatz[0]}-{parameterSatz[1]}%  {parameterSatz[2]}C {parameterSatz[3]}C')




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


    fig1.savefig(saveFigsDirectory+"cyc_funktion6_0_100_0.33_0.33_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cyc_funktion6_0_100_0.33_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cyc_funktion6_0_100_1_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cyc_funktion6_0_100_1.67_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig5.savefig(saveFigsDirectory+"cyc_funktion6_10_100_0.33_0.33_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig6.savefig(saveFigsDirectory+"cyc_funktion6_10_100_0.33_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig7.savefig(saveFigsDirectory+"cyc_funktion6_10_100_1_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig8.savefig(saveFigsDirectory+"cyc_funktion6_10_100_1.67_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig9.savefig(saveFigsDirectory+"cyc_funktion6_10_90_0.33_0.33_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig10.savefig(saveFigsDirectory+"cyc_funktion6_10_90_0.33_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig11.savefig(saveFigsDirectory+"cyc_funktion6_10_90_1_1_fixed.png", format='png',  pad_inches=0, transparent=False)
    fig12.savefig(saveFigsDirectory+"cyc_funktion6_10_90_1.67_1_fixed.png", format='png',  pad_inches=0, transparent=False)


    plt.show()



def funktion6_expo(x,a0,a1,a2,a3):
    return 1-a0*np.exp(a1*dsoc)*np.exp(a2/age_temp)*np.power(x,a3)
def funktion6_min_expo(parameters):
    global uniqueZellen,df_alleZellen,age_temp,dsoc
    summeErrors=0
    print(".", end="")
    for cell in uniqueZellen:

        xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
        ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
        dsoc = df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0]
        age_temp = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_temp"].iloc[0]
        
        y_modellierte= funktion6_expo(xdata, *parameters)
        error = np.mean((y_modellierte-ydata)**2)
        summeErrors = summeErrors + error
    return summeErrors
def funktion6_minimize_expo(arrayOfPathToCsvFiles):
    global age_temp,uniqueZellen,df_alleZellen
    fileToSaveValues = r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\02_Arbeit\Grafiken\Cyclic\cyc_funktion6_expo.csv"
    listOfPrograms = []
    
    # save the age_temp and age_voltage as tuples

    colorPalette = {273.15:"blue",283.15: "green",298.15: "darkorange", 313.15: "red"}
    lineStyle = {273.15:":",283.15:"--",298.15:"-.", 313.15:"-"}
    # lineStyle = {0.42:":",2.1:"--",3.78:"-.", 4.2:"-"}
    fig1 = plt.figure("cyc_funktion6_0_100_0.33_0.33")
    fig2 = plt.figure("cyc_funktion6_0_100_0.33_1")
    fig3 = plt.figure("cyc_funktion6_0_100_1_1")
    fig4 = plt.figure("cyc_funktion6_0_100_1.67_1")
    fig5 = plt.figure("cyc_funktion6_10_100_0.33_0.33")
    fig6 = plt.figure("cyc_funktion6_10_100_0.33_1")
    fig7 = plt.figure("cyc_funktion6_10_100_1_1")
    fig8 = plt.figure("cyc_funktion6_10_100_1.67_1")
    fig9 = plt.figure("cyc_funktion6_10_90_0.33_0.33")
    fig10= plt.figure("cyc_funktion6_10_90_0.33_1")
    fig11= plt.figure("cyc_funktion6_10_90_1_1")
    fig12= plt.figure("cyc_funktion6_10_90_1.67_1")

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

        sns.lineplot(x = xdata, y = ydata, color = "grey", linestyle = lineStyle[age_temp], ax = parameterSatzFig[parameterSatz].gca(), alpha = 0.5)
        

        pattern = r"P\d{3}_\d"  # Regex pattern to match "_PXXX_X"
        match = re.search(pattern, pathToCell)
        if match:
            extracted_value = match.group()
            # ax.set_title(extracted_value)
            new_row = {'nameCell': extracted_value,
                       'xvalues': [xdata], 'yvalues': [ydata], 'midsoc':midsoc, 'dsoc':dsoc, 'age_chg_rate':age_chg_rate, 'age_dischg_rate':age_dischg_rate, 'age_temp':age_temp}
        new_df = pd.DataFrame(new_row, index=[0])

        df_alleZellen = pd.concat([df_alleZellen, new_df], ignore_index=True)
        print(f"Added {extracted_value}")

    df_alleZellen.to_csv(tempFilesDirectory+"alleZellenData.csv")
    uniqueZellen = df_alleZellen["nameCell"].unique()


    start = [0.00001,0.05,1200,0.6]
    bounds = [(0.000001,0.00002),(0.03,0.07),(1000,1300),(0.2,0.8)]
    startTime = time.time()
    print(f"Starting optimization funktion 6 expo {startTime}")
    result = optimize.minimize(funktion6_min_expo,x0 = start, bounds= bounds)
    # class Result():
    #     def __init__(self):
    #         self.x = [-3.000e-01, -1.000e+00,  1.911e-03,  2.863e-02,  3.218e-01]
    # result = Result()
    print("")
    print( result)
    print(*result.x)
    endTime = time.time()
    print(f"Time elapsed: {endTime-startTime} seconds")

    tempArray = [273.15, 283.15, 298.15, 313.15]
    errorList = []
    globalErrorList = []
    for parameterSatz in listOfPrograms:
        errorList=[]
        parameterSatzFig[parameterSatz].gca().set_ylim(0.75, 1.05)
        for temperature in tempArray:
            age_temp = temperature
            dsoc = parameterSatz[1]-parameterSatz[0]
            age_chg_rate = parameterSatz[2]
            age_dischg_rate =parameterSatz[3]
            xdata = np.linspace(0,2000,12)
            sns.lineplot(x = xdata, y = funktion6_expo(xdata, *result.x), color = colorPalette[age_temp], ax = parameterSatzFig[parameterSatz].gca(),linestyle = lineStyle[age_temp])

        # Calculate error
        for cell in uniqueZellen:
            if (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_chg_rate"].iloc[0] == age_chg_rate) & (df_alleZellen[df_alleZellen["nameCell"]==cell]["age_dischg_rate"].iloc[0] == age_dischg_rate)& (df_alleZellen[df_alleZellen["nameCell"]==cell]["dsoc"].iloc[0] == dsoc):
                xdata = df_alleZellen[df_alleZellen["nameCell"]==cell]["xvalues"].iloc[0]
                ydata = df_alleZellen[df_alleZellen["nameCell"]==cell]["yvalues"].iloc[0]
                ydata = ydata[ydata>0.8]
                xdata = xdata[:len(ydata)]
                # age_voltage = df_alleZellen[df_alleZellen["nameCell"]==cell]["age_voltage"].iloc[0]
                
                y_modellierte= funktion6_expo(xdata, *result.x)
                error = (y_modellierte-ydata)**2
                errorList.append(np.mean(error*1000))
                globalErrorList.append(np.mean(error*1000))
        # print(f"RSME SOC age_voltage: {rmseParameterSatz}")
        rmseParameterSatz = np.sqrt(np.mean(errorList))
        line1 = plt.Line2D([0], [0], color=colorPalette[273.15], lw=0, label ="Temperatur")
        line2 = plt.Line2D([0], [0], color=colorPalette[273.15], lw=2,linestyle=':', label ="0°C")
        line3 = plt.Line2D([0], [0], color=colorPalette[283.15], lw=2,linestyle='--', label ="10°C")
        line4 = plt.Line2D([0], [0], color=colorPalette[298.15], lw=2,linestyle='-.', label ="25°C")
        line5 = plt.Line2D([0], [0], color=colorPalette[313.15], lw=2,linestyle='-',label ="40°C")
        legend1 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4,line5], loc='lower left')

        line1 = plt.Line2D([0], [0], color="grey", lw=2, label ="Messungen")
        line2 = plt.Line2D([0], [0], color=colorPalette[age_temp], lw=2, label = "Modellierung")

        exponentString = "%.3f" % result.x[3]
        rmseString = "%.3f" % rmseParameterSatz
        line3 = plt.Line2D([0], [0], color="grey", lw=0, label =f" Exponent: {exponentString}")
        line4 = plt.Line2D([0], [0], color="grey", lw=0, label =f" RMSE: {rmseString}")
        legend2 = parameterSatzFig[parameterSatz].gca().legend(handles=[line1, line2, line3,line4], loc='upper right')
        parameterSatzFig[parameterSatz].gca().add_artist(legend1)
        parameterSatzFig[parameterSatz].gca().add_artist(legend2)


        parameterSatzFig[parameterSatz].gca().grid(color='lightgrey', linestyle='--')
        parameterSatzFig[parameterSatz].gca().set_xlabel(f'EFC (-)')
        parameterSatzFig[parameterSatz].gca().set_ylabel('Relative Kapazität C(t)/C_init (-)')
        parameterSatzFig[parameterSatz].gca().set_title(f'Zyklische Modellierung - Funktion Nr. 6 - {parameterSatz[0]}-{parameterSatz[1]}%  {parameterSatz[2]}C {parameterSatz[3]}C')




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


    fig1.savefig(saveFigsDirectory+"cyc_funktion6_0_100_0.33_0.33.png", format='png',  pad_inches=0, transparent=False)
    fig2.savefig(saveFigsDirectory+"cyc_funktion6_0_100_0.33_1.png", format='png',  pad_inches=0, transparent=False)
    fig3.savefig(saveFigsDirectory+"cyc_funktion6_0_100_1_1.png", format='png',  pad_inches=0, transparent=False)
    fig4.savefig(saveFigsDirectory+"cyc_funktion6_0_100_1.67_1.png", format='png',  pad_inches=0, transparent=False)
    fig5.savefig(saveFigsDirectory+"cyc_funktion6_10_100_0.33_0.33.png", format='png',  pad_inches=0, transparent=False)
    fig6.savefig(saveFigsDirectory+"cyc_funktion6_10_100_0.33_1.png", format='png',  pad_inches=0, transparent=False)
    fig7.savefig(saveFigsDirectory+"cyc_funktion6_10_100_1_1.png", format='png',  pad_inches=0, transparent=False)
    fig8.savefig(saveFigsDirectory+"cyc_funktion6_10_100_1.67_1.png", format='png',  pad_inches=0, transparent=False)
    fig9.savefig(saveFigsDirectory+"cyc_funktion6_10_90_0.33_0.33.png", format='png',  pad_inches=0, transparent=False)
    fig10.savefig(saveFigsDirectory+"cyc_funktion6_10_90_0.33_1.png", format='png',  pad_inches=0, transparent=False)
    fig11.savefig(saveFigsDirectory+"cyc_funktion6_10_90_1_1.png", format='png',  pad_inches=0, transparent=False)
    fig12.savefig(saveFigsDirectory+"cyc_funktion6_10_90_1.67_1.png", format='png',  pad_inches=0, transparent=False)


    plt.show()




if __name__ == '__main__':
    funktion7_minimize([
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P064_3_S19_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P017_1_S01_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P017_2_S04_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P017_3_S05_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P018_1_S01_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P018_2_S02_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P018_3_S04_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P019_1_S01_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P019_2_S02_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P019_3_S05_C02.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P020_1_S01_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P020_2_S02_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P020_3_S03_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P021_1_S02_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P021_2_S03_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P021_3_S04_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P022_1_S03_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P022_2_S04_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P022_3_S05_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P023_1_S03_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P023_2_S04_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P023_3_S05_C03.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P024_1_S01_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P024_2_S04_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P024_3_S05_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P025_1_S01_C09.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P025_2_S02_C09.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P025_3_S03_C09.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P026_1_S01_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P026_2_S02_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P026_3_S03_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P027_1_S01_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P027_2_S02_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P027_3_S03_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P028_1_S02_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P028_2_S03_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P028_3_S04_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P029_1_S06_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P029_2_S08_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P029_3_S09_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P030_1_S06_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P030_2_S08_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P030_3_S09_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P031_1_S06_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P031_2_S07_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P031_3_S09_C03.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P032_1_S06_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P032_2_S07_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P032_3_S08_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P033_1_S07_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P033_2_S08_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P033_3_S09_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P034_1_S07_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P034_2_S08_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P034_3_S09_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P035_1_S08_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P035_2_S09_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P035_3_S10_C02.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P036_1_S05_C09.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P036_2_S09_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P036_3_S10_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P037_1_S05_C11.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P037_2_S06_C09.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P037_3_S10_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P038_1_S06_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P038_2_S07_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P038_3_S10_C03.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P039_1_S05_C10.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P039_2_S06_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P039_3_S07_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P040_1_S06_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P040_2_S07_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P040_3_S08_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P041_1_S11_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P041_2_S13_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P041_3_S14_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P042_1_S11_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P042_2_S13_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P042_3_S14_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P043_1_S11_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P043_2_S12_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P043_3_S14_C03.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P044_1_S11_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P044_2_S12_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P044_3_S13_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P045_1_S12_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P045_2_S13_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P045_3_S14_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P046_1_S12_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P046_2_S13_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P046_3_S14_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P047_1_S13_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P047_2_S14_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P047_3_S15_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P048_1_S10_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P048_2_S14_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P048_3_S15_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P049_1_S10_C10.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P049_2_S11_C09.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P049_3_S15_C02.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P050_1_S10_C09.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P050_2_S11_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P050_3_S12_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P051_1_S10_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P051_2_S11_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P051_3_S12_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P052_1_S11_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P052_2_S12_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P052_3_S13_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P053_1_S15_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P053_2_S16_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P053_3_S19_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P054_1_S16_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P054_2_S17_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P054_3_S19_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P055_1_S15_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P055_2_S16_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P055_3_S17_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P056_1_S16_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P056_2_S17_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P056_3_S18_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P057_1_S17_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P057_2_S18_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P057_3_S19_C08.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P058_1_S15_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P058_2_S18_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P058_3_S19_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P059_1_S15_C06.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P059_2_S18_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P059_3_S19_C04.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P060_1_S15_C03.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P060_2_S16_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P060_3_S19_C00.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P061_1_S16_C09.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P061_2_S17_C09.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P061_3_S18_C09.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P062_1_S16_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P062_2_S17_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P062_3_S18_C07.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P063_1_S16_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P063_2_S17_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P063_3_S18_C05.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P064_1_S17_C01.csv",
r"C:\Users\Yannick\bwSyncShare\MA Yannick Fritsch\00_Daten\prepr_res_run_7\cell_eocv2_P064_2_S18_C01.csv"
])