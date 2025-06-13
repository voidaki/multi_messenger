"""
author:voidaki
"""
import numpy as np
from data_loading import load_gravitational_wave_data, load_neutrino_data
from tqdm import tqdm
from skymap import HealPixSkymap
from scipy.integrate import nquad
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path
from likelihood import Paeffe

from utils import (
    t_overlap,
    IceCubeLIGO,
    IceCubeNeutrino,
    expnu,
    search_parameters,
    PEnu,
    temporal,
    Pr,
    sky_dist,
    PMgw,
    match_far,
    Aeff
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

def Aeff_skymap(epsilon, search_params, processes=8):
    import numpy as np
    import healpy as hp
    nside = 256
    npix = hp.nside2npix(nside)
    dec = hp.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)[1]
    args_list = [(epsilon, d, search_params) for d in dec]

    with Pool(processes=processes) as pool:
        effective_area = pool.map(Aeff_worker, args_list)

    effective_area = np.array(effective_area)
    return effective_area


def effective_area_skymap_generator():
    # runtime ~an hour
    epsilon_vals = [10**(epsilon+0.1) for epsilon in epsilon_dict()[:-1]]

    for i in tqdm(range(len(epsilon_vals))):
        if Path(f"effective_area{i}.npy").is_file():
            print(f"effective_area{i}.npy is already computed and saved")
            continue
        area_effective = Aeff_skymap(epsilon_vals[i], search_params=search_parameters("bns"))
        np.save(f"effective_area{i}.npy", area_effective)
        np.savetxt(f'effective_area{i}.csv', area_effective, delimiter=',')
        print(f"Effective Area skymap for the epsilon {epsilon_vals[i]} completed!")
        
        Skymap = HealPixSkymap(area_effective, moc=False)
        Skymap.plot()
        plt.show()

# effective_area_skymap_generator()

def allsky_aeff_integral_all_epsilon():
    # Returns 2.1242380368322262
    from pathlib import Path
    import astropy.units as u
    aeff_directory = Path("../data/neutrino_data/aeff_skymaps")

    aeff_integrals = []
    for i in range(41):
        filepath = aeff_directory / f"effective_area{i}.npy"
        epsilon = 10.0**epsilon_dict()[i]
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
    ndotgw_true = 1000 # Gpc^-3 * y^-1 for bns
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

def ndotnu(search_params=search_parameters("bns")): # Newer version, corrected
    """Returns the observable neutrino rate, calculated in 
    the integral in eq. (29).
    
    Parameters
    ----------
    search_params: Constant parameters for the model."""
    # ndotnu: (1.2822752736056162, 0.00023484038034700394)

    from utils import expnu_new
    ndotnu_true = 250.
    def P(r, Enu):
        return Pr(r, search_params)*PEnu(Enu, search_params)*(search_params.tnuplus-search_params.tnuminus)**-1*sky_dist()
    
    def integrant(r, Enu, dec):
        return P(r, Enu)*expnu_new(r, Enu, dec, search_params)*ndotnu_true

    result, error = nquad(integrant, [(np.min(distance_source), np.max(distance_source)), (search_params.Enumin, search_params.Enumax), (-90., 90.)])
    
    return result*4*np.pi, error

# print(f"ndotnu: {ndotnu()}")

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
    # yields 2922.2370310967326 for Enu=10^51 too big
    from utils import Pgw_bns_r, expnu_new

    ndotgwnu_true = 250.

    def P(r):
        return Pr(r, search_params)*Pgw_bns_r(r)*(search_params.tnuplus-search_params.tnuminus)*sky_dist()
    
    def integrant(dec, r):
        return P(r)*(1 - np.exp(-expnu_new(r, 1.e51, dec, search_params)))*ndotgwnu_true 
    
    rvals = np.linspace(5.0, 700.0, 1000)
    decvals = np.linspace(-90., 90., 100)

    dr = (700. - 5.)/1000.
    ddec = 180.0/100.
    integral_vals = []
    i=0
    for r in rvals:
        P_r = P(r)
        i += 1
        print(P_r, f"Percentage: {i/10}")
        for dec in decvals:
            integral_vals.append(P_r*(1.0 - np.exp(-expnu_new(r, 1.e46, dec, search_params))) * ndotgwnu_true)

    return np.sum(integral_vals)*dr*ddec
   
# print(f"ndotgwnu: {ndotgwnu()}")

search_params=search_parameters("bns")
# from utils import Aeff

# import numpy as np
# import matplotlib.pyplot as plt


# df = nu_data["effective_areas"]["IC86_II_effectiveArea"]

# # dec_vals = np.linspace(-90.0, 90.0, 100)
# # for dec in dec_vals:
# #     print(dec, ": ", Aeff(5.54, dec, search_params=search_parameters("bns"))) 

# from likelihood.neutrino import Paeffe
# from scipy.integrate import nquad

# # result, _ = nquad(Paeffe, [(search_params.epsilonmin, search_params.epsilonmax), (-89.9, 89.9)])
# # print("integral is:", result, "error is:", _)
# # print("ndotgwnu Result and error is: ", ndotgwnu())

# from data_loading import retrieve_event
# from skymap import *
# import matplotlib.pyplot as plt
# import hpmoc
# from utils import IceCubeNeutrino

# # skymap, tgw, far = retrieve_event('S250326y')
from astropy.time import Time

# # neutrino_list = [IceCubeNeutrino(Time(tgw-1.1, format='gps').mjd, 98.54, 54.1, 0.5, 4.2*10**4.2), 
# #                  IceCubeNeutrino(Time(tgw+1.3, format='gps').mjd, 101.2, 62.9, 0.8, 3.3*10**5.5)]

# # gw_skymap = HealPixSkymap.readQtable(skymap)
# # print(gw_skymap.to_table())
# # full_skymap = gw_skymap.rasterize(as_skymap=True)
# # # full_skymap.plot(neutrino_list=neutrino_list)
# # # plt.show()
# # print(full_skymap.to_table())
# # print("all sky integral is: ", full_skymap.allsky_integral())
# # print(full_skymap.nside, full_skymap.nside2ang(), full_skymap.pixels)
# # nu = full_skymap.neutrinoskymap(31.5, -41.43, 0.5)
# # nu_skymap = HealPixSkymap(nu.s, uniq=nu.u).rasterize(pad=0, as_skymap=True)

# # print(nu_skymap.to_table())

from coincidence_sig import TS, p_value, test_statistic
from data_loading import retrieve_event

# # cwb_skymap = retrieve_event("S250201al")[0]
# # print(cwb_skymap)

from pathlib import Path

EMP_NU = np.load("/home/aki/snakepit/multi_messenger_astro/data/neutrino_data/emprical_neutrinos.npy") # each element is np.array(epsilon, ra, dec, sigma)

LVK_skymap_folders = [Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4a"),
                      Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4b"),
                      Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4c")]

LVK_skymap_paths = [file for folder in LVK_skymap_folders
                    for file in folder.iterdir()]

false_alarm_rate_path = Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4_fars")

time_gps = Time('2025-01-01T00:00:00', scale='utc').gps
time_mjd = Time('2025-01-01T00:00:00', scale='utc').mjd

def gw_event():
    import astropy.units as u
    # gw_skymap_path = Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4a/S230627c_Bilby.multiorder.fits") # 1e-26
    gw_skymap_path = Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4a/S230615av_bayestar.multiorder.fits")
    skymap = HealPixSkymap.load_locally(gw_skymap_path, burst=False).rasterize(as_skymap=True)
    graceid = gw_skymap_path.name.split('_')[0]
    far_path = false_alarm_rate_path / (graceid + "_far.npy")
    far = np.load(far_path)
    neutrino_list = [IceCubeNeutrino(time_mjd, 240.0, -15.0, 1.2, 10**4.5)]
    rvals = np.linspace(0.0, 23000, 1000)
    return TS(time_gps, skymap, far, neutrino_list)

# test_stat = gw_event()
# print(test_stat)

import glob, os

from utils import Aeff, expnu, expnu_new
from skymap import Aeff_skymap
from scipy.integrate import quad,dblquad
from scipy.stats import poisson

def expected_neutrinos(dec, search_params=search_params):
    epsilon_bins = np.array(epsilon_dict())  # log10(E/GeV), e.g. [2.0, 2.2, 2.4, ..., 8.0]
    epsilon_vals = 10**((epsilon_bins[:-1] + epsilon_bins[1:]) / 2)  # bin centers in GeV
    delta_eps = 10**epsilon_bins[1:] - 10**epsilon_bins[:-1]  # bin widths in GeV

    aeff_vals = np.array([Aeff(eps, dec, search_params) for eps in epsilon_vals])
    integrand_vals = aeff_vals * epsilon_vals**-2

    integral = np.sum(integrand_vals * delta_eps)
    return integral

def integrant(r, dec):
    Enu = 10.**49
    return poisson.pmf(3.0, expnu_new(r, Enu, dec, search_params))

def PNnu(gw_skymap, Nnu, search_params=search_params):
    Mpc_to_cm = 3.085677581e24
    erg_to_GeV = 624.1509074
    mu_fraction = 1.0/3.0
    Enu = 10.**49

    filepath = "/home/aki/snakepit/multi_messenger_astro/core/expnu_dec.npy"
    data =  np.load(filepath, allow_pickle=True).item()
    dec_bins = data["dec_bins"]
    integral_vals = data["integrals"]

    def expnu_dec(dec_array):
        flat_dec = dec_array.flatten()
        idxs = [np.abs(dec_bins - d).argmin() for d in flat_dec]
        result = np.array([integral_vals[i] for i in idxs])
        return result.reshape(dec_array.shape)
    
    dec_vals = np.array((dec_bins[:-1] + dec_bins[1:]) / 2)
    delta_dec = (dec_bins[1:] - dec_bins[:-1])[0]

    r_vals = np.linspace(1.0, 700., 500)
    delta_r = float((700.0-1.0)/500)

    def inner(r, dec):
        return Enu*erg_to_GeV*search_params.fb*mu_fraction/(4*np.pi*np.log(search_params.epsilonmax/search_params.epsilonmin))*Mpc_to_cm**-2*r**-2*expnu_dec(dec)
    def integrant(dec, r):
        return poisson.pmf(Nnu, inner(r, dec))

    dec_grid, r_grid = np.meshgrid(dec_vals, r_vals, indexing='ij')
    integral_vals = integrant(dec_grid, r_grid)
    print(np.sum(integral_vals))
    total = np.sum(integral_vals)*delta_dec*delta_r
    return total

from utils import expnu_new, Aeff
decvals = np.linspace(-90., 90., 180)
ddec = np.deg2rad(decvals[1] - decvals[0]) 
expnus = [expnu_new(100., 1.e51, dec, search_params)*np.cos(np.deg2rad(dec)) for dec in decvals]

print(np.sum(expnus)*ddec/2)
plt.plot(decvals, expnus)
plt.show()

Mpc_to_cm = 3.085677581e24
erg_to_GeV = 624.1509074
Enu = 1.e51*erg_to_GeV
r = 100.*Mpc_to_cm
epsilon_bins = np.array(epsilon_dict()) 
epsilon_vals = 10**((epsilon_bins[:-1] + epsilon_bins[1:]) / 2)
delta_eps = 10**epsilon_bins[1:] - 10**epsilon_bins[:-1]
Aeff_vals = [Aeff(epsilon, 0.0, search_params)*epsilon**-2 for epsilon in epsilon_vals ]
int_vals = Aeff_vals*delta_eps*Enu/(4.0*np.pi)/13.8*r**-2
print(np.sum(int_vals))
plt.plot(epsilon_vals ,Aeff_vals)   
plt.xscale('log')
plt.yscale('log')
plt.show()

print("Null statistics plotting, p-value testing")
print(">-----------------------------------------<")

directory = '/home/aki/snakepit/multi_messenger_astro/core/noncwb'
teststat_dir = '/home/aki/snakepit/multi_messenger_astro/core/testnoncwb'

file_list = sorted(glob.glob(os.path.join(directory, '*.npy')))
test_files = sorted(glob.glob(os.path.join(directory, '*.npy')))

# Load and concatenate
all_null_stats = np.concatenate([np.load(f) for f in file_list])
all_null_stats = all_null_stats[all_null_stats <= 1.0]
# all_null_stats = all_null_stats[all_null_stats != 0]
all_test_stats = np.concatenate([np.load(f) for f in test_files])
# all_test_stats = all_test_stats[all_test_stats != 0]

print(f"Loaded {len(file_list)} files.")
print(f"Total null statistics: {len(all_null_stats)}")

threshold = 1e-40
null_stats_thresholded = np.where(all_null_stats < threshold, 0, all_null_stats)
epsilon = 1e-40
null_stats_clipped = np.clip(null_stats_thresholded, epsilon, None)
print(null_stats_clipped)

n_zeros = np.sum(null_stats_clipped == 0)
print(f"Number of zero statistics (≤ 1e-40): {n_zeros}")

log_null_stats = -np.log10(null_stats_clipped)
print(log_null_stats)

# Plot histogram
import matplotlib.pyplot as plt

p_values = np.array([p_value(test_stat, all_null_stats) for test_stat in all_test_stats])

bins = np.logspace(-40, -18, 50)

plt.figure(figsize=(8, 5))
# plt.hist(null_stats_clipped, bins=bins, color='goldenrod', edgecolor='black')
plt.hist(p_values, bins=20, color='red', edgecolor='black')
# plt.xlabel("Test Statistic")
plt.ylabel('Frequency')
# plt.xscale('log')
plt.title('Distribution of p-values for random events')
plt.xlabel('p-value')
plt.ylabel('Frequency')
# plt.title('Distribution of Null Statistics')
plt.grid(True)
plt.show()
