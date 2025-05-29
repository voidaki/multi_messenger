import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from utils import search_parameters
from likelihood import Paeffe


def Paeffe_plot():
    p=search_parameters("bns")
    epsilon_values = np.linspace(2.0, np.log10(p.epsilonmax), 150)
    declination_values = np.linspace(-90., 90., 150)

    epsilon_grid, declination_grid = np.meshgrid(epsilon_values, declination_values)
    Aeff_vec = np.vectorize(Paeffe)

    Aeff_values = Aeff_vec(epsilon_grid, declination_grid)
    print(f"Sum of Aeff values = {np.sum(Aeff_values)}")
    plt.figure(figsize=(8, 6))
    plt.imshow(Aeff_values, 
            extent=[epsilon_values.min(), epsilon_values.max(), 
                        declination_values.min(), declination_values.max()],
                origin="lower",
            aspect="auto",
            cmap="plasma",
            norm=LogNorm(vmin=np.nanmin(Aeff_values[Aeff_values > 0]), 
                            vmax=np.nanmax(Aeff_values)))

    plt.colorbar(label="Aeff*1/ϵ^2")
    plt.xlabel("Energy (log10(GeV))")
    plt.ylabel("Declination (degrees)")
    plt.title("Energy and sky location distribution from IceCube's Effective Area")

    plt.show()

def Pgw_bns_r(r):
    """
    Returns the probability distribution of distance using the O3 Sensitiity Measurements
    dataset. Only including the events with false alarm rate below 2 per day since LVK only 
    gets the measurements of them.
    """
    import matplotlib.pyplot as plt
    subthreshold_events = np.logical_or((far_gstlal <= 2), (far_mbta <= 2), (far_pycbc_hyperbank <= 2))
    
    distance_filtered = distance_source[subthreshold_events]
    plt.hist(distance_filtered, bins=30, color='darkorchid', edgecolor='black')
    plt.title("Prior Distance Distribution of Binary Neutron Star Mergers")
    plt.xlabel("Distance (Mpc)")
    plt.ylabel("Frequency")
    plt.show()

Pgw_bns_r(10)
