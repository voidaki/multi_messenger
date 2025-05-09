import numpy as np
from data_loading import load_gravitational_wave_data, load_neutrino_data
from tqdm import tqdm
from scipy.integrate import nquad

from likelihood import Paeffe

from utils import (
    t_overlap,
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
 
from utils import epsilon_dict
from multiprocessing import Pool

def Aeff_worker(args):
    epsilon, dec, search_params = args
    return Aeff(epsilon, dec, search_params)

def plot_effective_area_map(effective_area, title='Effective Area Skymap'):
    hp.mollview(
        effective_area,
        coord='C',  # 'C' = equatorial coordinates
        unit='cm²/GeV²',
        title=title,
        norm=None,
        cmap='viridis'
    )
    hp.graticule()
    plt.show()

def Aeff_skymap(epsilon, skymap, search_params, processes=8):
    import numpy as np
    dec = skymap.nside2ang()[1]
    args_list = [(epsilon, d, search_params) for d in dec]

    with Pool(processes=processes) as pool:
        effective_area = pool.map(Aeff_worker, args_list)

    effective_area = np.array(effective_area)
    return effective_area


def effective_area_skymap_generator():
    # runtime ~an hour
    epsilon_vals = [10**epsilon for epsilon in epsilon_dict()]

    for i in range(len(epsilon_vals)):
        area_effective = Aeff_skymap(epsilon_vals[i], full_skymap, search_params=search_parameters("bns"))
        np.save(f"effective_area{i}.npy", area_effective)
        np.savetxt(f'effective_area{i}.csv', area_effective, delimiter=',')
        print(f"Effective Area skymap for the epsilon {epsilon_vals[i]} completed!")
        if i in [0, 5, 10, 15]:
            plot_effective_area_map(area_effective)

def allsky_aeff_integral_all_epsilon():
    # Returns 2.1242380368322262
    from pathlib import Path
    aeff_directory = Path("../data/neutrino_data/aeff_skymaps")

    aeff_integrals = []
    for i in range(41):
        filepath = aeff_directory / f"effective_area{i}.npy"
        epsilon = 10**epsilon_dict()[i]
        s = np.load(filepath)*epsilon**-2
        AeffSkymap = HealPixSkymap(s*1/u.sr, moc=False)
        integral = AeffSkymap.allsky_integral()
        aeff_integrals.append(integral)
        print(integral)

    print(sum(aeff_integrals)*4*np.pi)

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


# def ndotgwnu(search_params=search_parameters("bns")):
#     """Returns the joint observable GW and neutrino rate.
    
#     Parameters
#     ----------
#     search_params: Constant parameters for the model."""
#     from scipy.integrate import quad, nquad
#     from scipy.interpolate import RegularGridInterpolator
    
#     ndotgwnu_true = 1000.
    
#     def Pgw(r):
#         """Histogram of the O3-sensitivity estimates injections.
        
#         Parameters
#         ----------
#         r: float
#             Distance of the gravitational wave event, in Mpc
#         """
#         subthresholds = np.logical_or((far_gstlal <= 2), (far_mbta <= 2), (far_pycbc_hyperbank <= 2))
#         rsubthreshold = distance_source[subthresholds]
        
#         counts, bin_edges = np.histogram(rsubthreshold, bins=100)

#         if r < bin_edges[0] or r > bin_edges[-1]:
#             return 0
    
#         bin_i = np.digitize(r, bin_edges) - 1

#         return counts[bin_i]/len(distance_source)
    
#     def Pnu(r, Enu, theta):
#         return quad(lambda epsilon: Paeffe(epsilon, theta), np.log10(search_params.epsilonmin), np.log10(search_params.epsilonmax), limit=200)[0]*Enu*r**-2
#     print(Pnu(400.90, 0.5*10**49, 51.25))
#     def integrant(r, Enu, theta):
#         return r**2*np.sin(theta)*Pgw(r)*Pnu(r, Enu, theta)
    
#     result, error = nquad(integrant, [
#         (2.0, 730.0),
#         (search_params.Enumin, search_params.Enumax),
#         (0.0, np.pi)
#     ], opts=[{'limit': 200}, {'limit': 200}, {'limit': 200}])


#     return result*ndotgwnu_true*2*np.pi*4*np.pi, error

def ndotgwnu(search_params=search_parameters("bns")):
    """Returns the joint observable GW and neutrino rate.
    
    Parameters
    ----------
    search_params: Constant parameters for the model."""
    from scipy.integrate import nquad
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
    
    def integrant(r, Enu, theta, phi):
        return r**2*Pgw(r)*np.sin(theta)*(1-np.exp(-expnu(r, Enu, search_params)))
    
    return nquad(integrant, [(np.min(distance_source), np.max(distance_source)), (search_params.Enumin, search_params.Enumax), (0, 2*np.pi), (0, np.pi)])
        

search_params=search_parameters("bns")
from utils import Aeff

import numpy as np
import matplotlib.pyplot as plt


df = nu_data["effective_areas"]["IC86_II_effectiveArea"]

# dec_vals = np.linspace(-90.0, 90.0, 100)
# for dec in dec_vals:
#     print(dec, ": ", Aeff(5.54, dec, search_params=search_parameters("bns"))) 

from likelihood.neutrino import Paeffe
from scipy.integrate import nquad

# result, _ = nquad(Paeffe, [(search_params.epsilonmin, search_params.epsilonmax), (-89.9, 89.9)])
# print("integral is:", result, "error is:", _)
# print("ndotgwnu Result and error is: ", ndotgwnu())

from data_loading import retrieve_event
from skymap import *
import matplotlib.pyplot as plt
import hpmoc
from utils import IceCubeNeutrino

skymap, tgw, far = retrieve_event('S250326y')
from astropy.time import Time

neutrino_list = [IceCubeNeutrino(Time(tgw-1.1, format='gps').mjd, -57.08, -29.42, 0.5, 4.2*10**4), 
                 IceCubeNeutrino(Time(tgw+1.3, format='gps').mjd, -67.1, -19.8, 0.8, 3.3*10**5)]

gw_skymap = HealPixSkymap.readQtable(skymap)
print(gw_skymap.to_table())
full_skymap = gw_skymap.rasterize(as_skymap=True)
# full_skymap.plot(neutrino_list=neutrino_list)
# plt.show()
print(full_skymap.to_table())
print("all sky integral is: ", full_skymap.allsky_integral())
print(full_skymap.nside, full_skymap.nside2ang(), full_skymap.pixels)
nu = full_skymap.neutrinoskymap(31.5, -41.43, 0.5)
nu_skymap = HealPixSkymap(nu.s, uniq=nu.u).rasterize(pad=0, as_skymap=True)

print(nu_skymap.to_table())
import healpy as hp
import astropy.units as u
def skymap_integral(gwskymap, neutrino_list):
    pix_area = hp.nside2pixarea(gwskymap.nside)*u.sr
    nuskymap = emptyskymap(0.0, gwskymap)
    for neutrino in neutrino_list:
        a = gwskymap.neutrinoskymap(neutrino.ra, neutrino.dec, neutrino.sigma)
        a = HealPixSkymap(a.s, uniq=a.u).rasterize(pad=0., as_skymap=True)
        nuskymap.pixels += (a.pixels*pix_area).to(u.dimensionless_unscaled).value
    prob_dens = gwskymap.pixels*nuskymap.pixels
    prob_map = (prob_dens*pix_area).to(u.dimensionless_unscaled).value
    return prob_map.sum()

from scipy.stats import poisson
from coincidence_sig import TS

full_skymap.plot(neutrino_list)
plt.show()
print(TS(tgw, full_skymap, far, neutrino_list))
