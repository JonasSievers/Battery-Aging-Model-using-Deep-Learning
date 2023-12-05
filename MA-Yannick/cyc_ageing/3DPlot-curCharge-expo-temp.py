import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from math import ceil
from scipy.interpolate import griddata
ax = plt.axes(projection='3d')
#  Current 0.33
x = np.linspace(0.33,0.33,4) # CURRENT
y = np.linspace(0, 40, 4) # TEMP
z = [0.763459, 0.554453,0.470829,0.554923]

ax.plot3D(x, y, z, 'blue',linewidth=2.0)

#  Current 1
x = np.linspace(1,1,4) # CURRENT
y = np.linspace(0, 40, 4) # TEMP
z = [1,0.870317,0.649020,0.696886]
ax.plot3D(x, y, z, 'blue',   linewidth=2.0)#linestyle="dashed",

#  Current 1.67
x = np.linspace(1.67,1.67,4) # CURRENT
y = np.linspace(0, 40, 4) # TEMP
z = [1,0.941235,0.918361,0.726903]

ax.plot3D(x, y, z, 'darkblue',linewidth=3.0)






# Calculate array of format 11x4

array = np.zeros((20,4))
# print(array)
# datenLinien = np.arange(0,anzahlLinien,anzahlZwischenLinien+1)
# print(datenLinien)
# ALLES HIER BEI 40°C

# X - C Rate charging : 0.33 -> 1.67 in 20 schritten
#  print(np.linspace(0.33,1.67,20))

# SOC  10      50      90 100
# Line 0 1 2 3 4 5 6 7 8  9
array[0] = [0.763459, 0.554453,0.470829,0.554923]
array[10] = [1,0.870317,0.649020,0.696886]
array[19] = [1,0.941235,0.918361,0.726903]


# # print(array)


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

    


transformed_list = np.array(list(zip(*array)))
print(transformed_list)











def function(x, y):
    print(x)
    print(y)
    print(x+y)
    return transformed_list



x = np.linspace(0.33,1.67,20) #charge rate
# x = np.array([10,20,30,40,50,60,70,80,90,100])

y = np.linspace(0, 40, 4) # Temp
print("x - before meshgrid",x)
print("y - before meshgrid",y)


X, Y = np.meshgrid(x, y)
Z = function(X, Y)


ax.plot_surface(X, Y, Z, cmap='autumn', alpha=0.9)

ax.set_title("Exponent Abhängigkeit von C-Rate und Temperatur")
ax.set_xlabel('x - C-Rate (-)', fontsize=12)
ax.set_ylabel('y - Temperatur (°C)', fontsize=12)
ax.set_zlabel('z - Exponent (-)', fontsize=12)

plt.show()
