import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import heapq

shots = pd.read_csv(sys.argv[1])
print shots.head()

index = shots.loc[0:].index

x = np.array(index)

x_values = shots.loc[0:]['x']
y_values = shots.loc[0:]['y']
z_values = shots.loc[0:]['z']

y_xs = np.array(x_values)
y_ys = np.array(y_values)
y_zs = np.array(z_values)

plt.plot(x,y_xs)
plt.show()

largest = heapq.nlargest(10, range(len(y_xs)), y_xs.take)
largest.sort()
largest.reverse()

i = 0
while i < len(largest):
    if i + 1 >= len(largest):
        break
    else:
        print "subtracting " + str(largest[i]) + "from " + str(largest[i + 1])
        if abs(largest[i] - largest[i + 1]) < 100:
            rem = largest[i+1]
            largest.remove(rem)
        else:
            i += 1
print largest
#plt.plot(y_xs,x)
#plt.show()
#plt.plot(x,y_ys)
#plt.show()
#plt.plot(x,y_zs)
#plt.show()

#plt.scatter(x,y_xs)
#plt.show()
