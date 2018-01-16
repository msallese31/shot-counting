import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema

def GetTime(autocorr):
    time = 0
    thresh = int(sys.argv[3])
    sortedauto = np.array(autocorr)
    print(sortedauto[0])
    sortedauto.sort()
    sortedauto = sortedauto[-50:-1]
    threshold = sortedauto.mean()
    print(threshold)
    times = []
    while time < len(autocorr):
        if autocorr[time] > threshold:
            print(time)
            times.append(time)
            time += 100
        time += 1
    #print(argrelextrema(autocorr, np.greater))
    plt.plot(autocorr)
    y_times = []
    for a in times:
        y_times.append(autocorr[a])
    print(times)
    print(y_times)

    plt.plot(times, y_times, 'ro')
    plt.show()



def main():
    data = pd.read_csv(sys.argv[1])
    pattern  = pd.read_csv(sys.argv[2])
    xs = data['x']
    ys = data['y']
    zs = data['z']

    xps = pattern['x']
    yps = pattern['y']
    zps = pattern['z']
    plt.plot(xs)

    plt.savefig("xs", ext="png", close=False, verbose=True)
    plt.clf()
    plt.plot(ys)
    plt.savefig("ys", ext="png", close=False, verbose=True)
    plt.clf()
    plt.plot(zs)
    plt.savefig("zs", ext="png", close=False, verbose=True)
    plt.clf()

    patt = xps[124:189]
    autocorr = np.correlate(xs, patt)
    GetTime(autocorr)

if __name__ == '__main__':
        main()

