import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from math import ceil
from scipy.interpolate import griddata
ax = plt.axes(projection='3d')
#  TEMP 40 SOC 10
x = np.linspace(10,10,11) # SOC
y = np.linspace(0, 192, 11) # Time
z = [1.0, 0.998094841946901, 0.992713117031652, 0.986352490781579, 0.979341998687388, 0.973943292839791, 0.9690165854096926, 0.9644320312294071, 0.9602290164537407, 0.9562714950276829, 0.9525940357685547]

ax.plot3D(x, y, z, 'blue',linewidth=2.0)

#  TEMP 0 SOC 10
x = np.linspace(10,10,11) # SOC
y = np.linspace(0, 192, 11) # Time
z = [1.0, 0.9980996379858981, 0.9943648474707425, 0.98956641960187, 0.9841493783867743, 0.9800734381775397, 0.9760090053776915, 0.9723434252347731, 0.9689491811177023, 0.9657586587615734, 0.9627513675823517]
ax.plot3D(x, y, z, 'blue', linestyle="dashed",  linewidth=2.0)


#  TEMP 40 SOC 50
x = np.linspace(50,50,11) # SOC
y = np.linspace(0, 192, 11) # Time
z = [1.0, 0.9919857426505261, 0.9810586888443487, 0.971950921840063, 0.9628797479938319, 0.9554099718795959, 0.9487400784108013, 0.9429481929853702, 0.937520321406974, 0.9324449776243203, 0.927594302775515]

ax.plot3D(x, y, z, 'darkblue',linewidth=3.0)


#  TEMP 40 SOC 90
x = np.linspace(90,90,11) # SOC
y = np.linspace(0, 192, 11) # Time
z = [1.0, 0.987435521778311, 0.9733968267575941, 0.9617191293400881, 0.950873597629936, 0.9418398238609006, 0.9341723378328762, 0.9271277915753405, 0.920685304716395, 0.9150347282951666, 0.9095884846265028]

ax.plot3D(x, y, z, 'orange',linewidth=3.0)

#  TEMP 0 SOC 90
x = np.linspace(90,90,11) # SOC
y = np.linspace(0, 192, 11) # Time
z = [1.0, 0.9905062949822253, 0.9828667310859057, 0.9756242789222701, 0.968422996805045, 0.9629641479090396, 0.9573195651446151, 0.9525204259478411, 0.9480468933968501, 0.9439561634145576, 0.9401580075046879]

ax.plot3D(x, y, z, 'orange',linestyle = "dashed",linewidth=3.0)


#  TEMP 40 SOC 100
x = np.linspace(100,100,11) # SOC
y = np.linspace(0, 192, 11) # Time
z = [1.0, 0.9944935010996315, 0.9843216236352178, 0.9742866932075708, 0.9645216945845633, 0.9566820634059449, 0.9494436214704484, 0.9429615400627368, 0.937085427698303, 0.9311999960075172, 0.9251499587639002]

ax.plot3D(x, y, z, 'red',linewidth=3.0)


#  TEMP 0 SOC 100
x = np.linspace(100,100,11) # SOC
y = np.linspace(0, 192, 11) # Time
z = [1.0, 0.9921999527447604, 0.9847430890824528, 0.9775723092949923, 0.9698463950918225, 0.9641775552986251, 0.9584319391599113, 0.9537343516549978, 0.9493149777818045, 0.9453465405764341, 0.9416513325181396]

ax.plot3D(x, y, z, 'red',linestyle = "dashed",linewidth=3.0)

# Calculate array of format 11x4

array = np.zeros((10,11))
# print(array)
# datenLinien = np.arange(0,anzahlLinien,anzahlZwischenLinien+1)
# print(datenLinien)
# ALLES HIER BEI 40°C
# SOC  10      50      90 100
# Line 0 1 2 3 4 5 6 7 8  9
array[0] = [1.0, 0.998094841946901, 0.992713117031652, 0.986352490781579, 0.979341998687388, 0.973943292839791, 0.9690165854096926, 0.9644320312294071, 0.9602290164537407, 0.9562714950276829, 0.9525940357685547]
array[4] = [1.0, 0.9919857426505261, 0.9810586888443487, 0.971950921840063, 0.9628797479938319, 0.9554099718795959, 0.9487400784108013, 0.9429481929853702, 0.937520321406974, 0.9324449776243203, 0.927594302775515]
array[8] = [1.0, 0.987435521778311, 0.9733968267575941, 0.9617191293400881, 0.950873597629936, 0.9418398238609006, 0.9341723378328762, 0.9271277915753405, 0.920685304716395, 0.9150347282951666, 0.9095884846265028]
array[9] = [1.0, 0.9944935010996315, 0.9843216236352178, 0.9742866932075708, 0.9645216945845633, 0.9566820634059449, 0.9494436214704484, 0.9429615400627368, 0.937085427698303, 0.9311999960075172, 0.9251499587639002]


# print(array)


for index in range(0,len(array)):
    if (array[index,0] == 0.0):
        currentRow = index
        nextRowWithValue = 0
        lastRowWithValue = 0
        # look for next row with a value
        for index2 in range(index,len(array)):
            if (array[index2,0] != 0.0):
                nextRowWithValue = index2
                break
        # look for next row with a value
        for index2 in range(0,len(array)):
            if (array[index-index2,0] != 0.0):
                lastRowWithValue = index-index2
                break
        print("currentRow",currentRow)
        print("nextRowWithValue",nextRowWithValue)
        print("lastRowWithValue",lastRowWithValue)
        anzahlZwischenLinien = nextRowWithValue - lastRowWithValue
        currentRowCycle = nextRowWithValue - index
        a = (array[lastRowWithValue]-array[nextRowWithValue])*(anzahlZwischenLinien)/(anzahlZwischenLinien+1)
        b = (array[lastRowWithValue]-array[nextRowWithValue])/(anzahlZwischenLinien+1)
        c = array[lastRowWithValue]
        interRowNumber = currentRowCycle%(anzahlZwischenLinien)
        b2 = interRowNumber*b
        ab2 =a-b2
        array[index] = c-ab2
        print(array[index])

print(array)

    

print(array)
transformed_list = np.array(list(zip(*array)))
print(transformed_list)











def function(x, y):
    print(x)
    print(y)
    print(x+y)
    return transformed_list
    return np.array([[  1 , 1, 1, 1],[ 0.9,0.9,0.9,0.9],[0.8,0.8,0.8,0.8]])


# x = np.linspace(0,100,3) # SOC
x = np.array([10,20,30,40,50,60,70,80,90,100])

y = np.linspace(0, 192, 11) # Time
print("x - before meshgrid",x)
print("y - before meshgrid",y)


X, Y = np.meshgrid(x, y)
Z = function(X, Y)


ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.3)

ax.set_title("Kapazität bei unterschiedlichen SOCs im Verlauf der Zeit")
ax.set_xlabel('x - SOC', fontsize=12)
ax.set_ylabel('y - Time (days)', fontsize=12)
ax.set_zlabel('z - Rel. Capacity', fontsize=12)

plt.show()
