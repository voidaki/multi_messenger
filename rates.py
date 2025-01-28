from multimessenger import *
import numpy as np
import matplotlib.pyplot as plt

# Neutrinos detection rate
times1 = np.empty(1)
for key in dataframes_events.keys():
    df = dataframes_events[key]
    days = df["MJD[days]"]
    days = np.array(days)
    times1 = np.concatenate((times1, days))
times = np.delete(times1, 0)
print((np.max(times) - np.min(times)) / 365.25)
rate = len(times) / ((np.max(times) - np.min(times)) * 24 * 60 * 60)
print(rate * 1000)
