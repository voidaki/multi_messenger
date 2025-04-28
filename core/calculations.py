import numpy as np
from data_loading import load_gravitational_wave_data, load_neutrino_data
from tqdm import tqdm
from scipy.integrate import nquad

from likelihood import Paeffe

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
    
    def integrant(r, Enu):
        return P(r, Enu)*expnu(r, Enu, search_params)*ndotnu_true*search_params.fb**-1

    result, error = nquad(integrant, [(np.min(distance_source), np.max(distance_source)), (search_params.Enumin, search_params.Enumax)])
    
    return result*4*np.pi, error


def ndotgwnu(search_params=search_parameters("bns")):
    """Returns the joint observable GW and neutrino rate.
    
    Parameters
    ----------
    search_params: Constant parameters for the model."""
    from scipy.integrate import quad, nquad
    from scipy.interpolate import RegularGridInterpolator
    
    ndotgwnu_true = 1000.
    
    def Pgw(r):
        """Histogram of the O3-sensitivity estimates injections.
        
        Parameters
        ----------
        r: float
            Distance of the gravitational wave event, in Mpc
        """
        subthresholds = np.logical_or((far_gstlal <= 2), (far_mbta <= 2), (far_pycbc_hyperbank <= 2))
        rsubthreshold = distance_source[subthresholds]
        
        counts, bin_edges = np.histogram(rsubthreshold, bins=100)

        if r < bin_edges[0] or r > bin_edges[-1]:
            return 0
    
        bin_i = np.digitize(r, bin_edges) - 1

        return counts[bin_i]/len(distance_source)
    
    def Pnu(r, Enu, theta):
        return quad(lambda epsilon: Paeffe(epsilon, theta), np.log10(search_params.epsilonmin), np.log10(search_params.epsilonmax), limit=200)[0]*Enu*r**-2
    print(Pnu(400.90, 0.5*10**49, 51.25))
    def integrant(r, Enu, theta):
        return r**2*np.sin(theta)*Pgw(r)*Pnu(r, Enu, theta)
    
    result, error = nquad(integrant, [
        (2.0, 730.0),
        (search_params.Enumin, search_params.Enumax),
        (0.0, np.pi)
    ], opts=[{'limit': 200}, {'limit': 200}, {'limit': 200}])


    return result*ndotgwnu_true*2*np.pi*4*np.pi, error


from data_loading import retrieve_event
from skymap import *
import matplotlib.pyplot as plt
import hpmoc

skymap, tgw, far = retrieve_event('S250326y')

gw_skymap = HealPixSkymap.readQtable(skymap)
print(gw_skymap.nside2area_per_pix())
full_skymap = gw_skymap.rasterize(as_skymap=True)
print(full_skymap)
print(full_skymap.nside, full_skymap.nside2ang(), full_skymap.pixels)
nu = full_skymap.neutrinoskymap(31.5, -41.43, 0.5)
print("\nThe neutrino skymap:", nu)