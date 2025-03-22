import numpy as np
from data_loading import load_gravitational_wave_data, load_neutrino_data
from tqdm import tqdm
from scipy.integrate import nquad

from utils import (
    IceCubeLIGO,
    expnu,
    search_parameters,
    PEnu,
    temporal,
    Pr,
    sky_dist,
    PMgw,
    match_far
)

gw_data = load_gravitational_wave_data()
gpstime_source = gw_data["gpstime"]
mass1_source = gw_data["mass1"]
mass2_source = gw_data["mass2"]
distance_source = gw_data["distance"]
inc_source = gw_data["inclination"]
ra_source = gw_data["right_ascension"]
dec_source = gw_data["declination"]
far_gstlal = gw_data["far_gstlal"]
far_mbta = gw_data["far_mbta"]
far_pycbc_hyperbank = gw_data["far_pycbc_hyperbank"]
far_pycbc_bbh = gw_data["far_pycbc_bbh"]
alt_source = gw_data["altitude"]
az_source = gw_data["azimuth"]

nu_data = load_neutrino_data()
dataframes_events = nu_data["events"]
 
def ndotgw(search_params=search_parameters("bns")):
    """Returns the observable GW rate, calculated in 
    the integral in eq. (26).
    
    Parameters
    ----------
    search_params: Constant parameters for the model."""
    # Returns 3.9255868248144146 per year
    # Check units
    far_mask = np.logical_or(far_gstlal <= 2, far_mbta <= 2, far_pycbc_hyperbank <= 2)
    gpstime_filtered = gpstime_source[far_mask]
    distance_filtered = distance_source[far_mask]
    mass1_filtered = mass1_source[far_mask]
    mass2_filtered = mass2_source[far_mask]
    
    prob = 0
    integral = 0
    ndotgw_true = 1000 # Gpc^-3 * y^-1
    T_obs = search_params.tgwplus - search_params.tgwminus
    for i in tqdm(range(len(gpstime_filtered)), desc="computing integral"):
        prob += temporal(T_obs)*Pr(distance_filtered[i], search_params)*sky_dist()*PMgw(mass1_filtered[i], mass2_filtered[i], search_params)
        integral += ndotgw_true*prob
    return integral*4*np.pi/len(gpstime_source)


def ndotnu(search_params=search_parameters("bns")):
    """Returns the observable neutrino rate, calculated in 
    the integral in eq. (29).
    
    Parameters
    ----------
    search_params: Constant parameters for the model."""
    # Returns 2.7402641832764836e-05
    # error = 2.42098891015745e-20
    ndotnu_true = 250.
    def P(r, Enu):
        return Pr(r, search_params)*PEnu(Enu, search_params)*temporal(search_params.tnuplus-search_params.tnuminus)*sky_dist()
    
    def integrand(r, Enu):
        return P(r, Enu)*expnu(r, Enu, search_params)*ndotnu_true*search_params.fb**-1

    result, error = nquad(integrand, [(np.min(distance_source), np.max(distance_source)), (search_params.Enumin, search_params.Enumax)])
    
    return result*4*np.pi, error


def ndotgwnu(search_params=search_parameters("bns")):
    """Returns the joint observable GW and neutrino rate.
    
    Parameters
    ----------
    search_params: Constant parameters for the model."""

    ndotgwnu_true = 1.0
    def P(r, Enu, M1, M2):
        return Pr(r, search_params)*PEnu(Enu, search_params)*temporal(search_params.tnuplus-search_params.tnuminus)*sky_dist()*PMgw(M1, M2, search_params)
   
    def integrand(r, Enu, M1, M2, gpstime, right_ascension, declination):
        if match_far(gpstime, r, right_ascension*180./np.pi, declination*180./np.pi, M1, M2) >= 2.0:
            return 0
        else:
            return P(r, Enu, M1, M2)*expnu(r, Enu, search_params)*1.0*search_params.fb**-1
    

    result, error = nquad(integrand, [(np.min(distance_source), np.max(distance_source)), 
                                      (search_params.Enumin, search_params.Enumax), 
                                      (search_params.Mgwmin, search_params.Mgwmax),
                                      (search_params.Mgwmin, search_params.Mgwmax),
                                      (np.min(gw_data["gpstime"]), np.max(gw_data["gpstime"])),
                                      (0.0, 2*np.pi),
                                      (-np.pi/2, np.pi/2)])
    
    return result, error


from likelihood import Paeffe
import matplotlib.pyplot as plt

Paeffe_vec = np.vectorize(Paeffe)
search_params = search_parameters("bns")
epsilonbins = np.linspace(np.log10(search_params.epsilonmin), np.log10(search_params.epsilonmax), 50)
decbins = np.linspace(-90.0, 90.0, 100)

X, Y = np.meshgrid(epsilonbins, decbins)

Z = Paeffe_vec(X, Y, search_params)

plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, Z, cmap="plasma", shading="auto")

# Add a colorbar
plt.colorbar(label="Function Value")

# Labels and title
plt.xlabel("Energy")
plt.ylabel("Declination Angle")
plt.title("2D Gaussian Function")

# Show the plot
plt.show()