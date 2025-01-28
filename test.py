from multimessenger import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

#print("Probability of false alarm rate with histogram method", P_far(1e-5))
#print("Probability of false alarm rate with integal method: ", P_signal_GW_F(1e-5))


P_Aeffvec = np.vectorize(P_skyloc_Aeff)
P_datavec = np.vectorize(P_skyloc_data)

# declination_vals = np.linspace(-89, 89, 500)
# epsilon_vals = [2, 3, 3.5, 4, 6]
# for epsilon in epsilon_vals:
#    p_data = P_datavec(declination_vals, epsilon)
#    p_Aeff = P_Aeffvec(declination_vals, epsilon)
#    p = p_data / p_Aeff
#    plt.plot(declination_vals, p_Aeff, label=f"Epsilon = {epsilon}")
# plt.legend()
# plt.show()

times1 = np.empty(1)
for key in dataframes_events.keys():
    df = dataframes_events[key]
    days = df["MJD[days]"]
    days = np.array(days)
    times1 = np.concatenate((times1, days))
times = np.delete(times1, 0)
print((np.max(times) - np.min(times)) / 365.25)
rate = len(times) / ((np.max(times) - np.min(times)) * 24 * 60 * 60)
print(rate)


epsilon_vals = [ 10] 
declination_vals = np.linspace(-89, 89, 500)
p_emprical = P_datavec(declination_vals, epsilon)
p_Aeff = P_Aeffvec(declination_vals, epsilon)
plt.plot(declination_vals, p_Aeff, label=f"{epsilon}")
#plt.plot(declination_vals, p_emprical)
plt.legend()
plt.show()
