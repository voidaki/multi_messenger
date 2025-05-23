import numpy as np
from data_loading import load_gravitational_wave_data, load_neutrino_data
from tqdm import tqdm
from scipy.integrate import nquad
import healpy as hp
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


def effective_area_skymap_generator(full_skymap):
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
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # subthresholds = np.logical_or((far_gstlal <= 2), (far_mbta <= 2), (far_pycbc_hyperbank <= 2))
    # rsubthreshold = distance_source[subthresholds]
        
    # counts, bin_edges = np.histogram(rsubthreshold, bins=100, range=(0, 700))

    def make_Pgw():
        subthresholds = np.logical_or((far_gstlal <= 2), (far_mbta <= 2), (far_pycbc_hyperbank <= 2))
        rsubthreshold = distance_source[subthresholds]
        
        counts, bin_edges = np.histogram(rsubthreshold, bins=100, range=(0, 700), density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        # Avoid log(0), NaNs
        counts = np.maximum(counts, 1e-10)

        return interp1d(bin_centers, counts, bounds_error=False, fill_value=0.0)

    Pgw_interp = make_Pgw()

    def Pgw(r):
        return Pgw_interp(r)

    # def Pgw(r):
    #     """Histogram of the O3-sensitivity estimates injections.
            
    #     Parameters
    #     ----------
    #     r: float
    #         Distance of the gravitational wave event, in Mpc
    #     """
    #     if r < bin_edges[0] or r > bin_edges[-1]:
    #         return 0
    
    #     bin_i = np.clip(np.digitize(r, bin_edges) - 1, 0, len(counts) - 1)

    #     return counts[bin_i]/len(distance_source)
    
    # def integrant(r, Enu, theta):
    #     return r**2*Pgw(r)*np.sin(theta)*(1-np.exp(-search_params.nu_51_100*(Enu/search_params.Enumax)*(100/r)**2))

    def integrand_vec(x):
        r, Enu, theta = x[:, 0], x[:, 1], x[:, 2]
        return r**2 * Pgw(r) * np.sin(theta) * (1 - np.exp(-expnu(r, Enu, search_params)))
    
    N = 100000
    r_samples = np.random.uniform(0, 700, N)
    enu_samples = np.random.uniform(search_params.Enumin, search_params.Enumax, N)
    theta_samples = np.random.uniform(0, 2*np.pi, N)

    samples = np.stack([r_samples, enu_samples, theta_samples], axis=-1)
    vals = integrand_vec(samples)

    volume = 700 * (search_params.Enumax - search_params.Enumin) * (2 * np.pi)
    integral = np.mean(vals) * volume
    print("Integral ≈", integral)
    # return nquad(integrant, [(4.5, 695.0), (search_params.Enumin, search_params.Enumax), (0.0, 2*np.pi)], opts=[{"limit": 200}])
        

# print(ndotgwnu())

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

# skymap, tgw, far = retrieve_event('S250326y')
from astropy.time import Time

# neutrino_list = [IceCubeNeutrino(Time(tgw-1.1, format='gps').mjd, 98.54, 54.1, 0.5, 4.2*10**4.2), 
#                  IceCubeNeutrino(Time(tgw+1.3, format='gps').mjd, 101.2, 62.9, 0.8, 3.3*10**5.5)]

# gw_skymap = HealPixSkymap.readQtable(skymap)
# print(gw_skymap.to_table())
# full_skymap = gw_skymap.rasterize(as_skymap=True)
# # full_skymap.plot(neutrino_list=neutrino_list)
# # plt.show()
# print(full_skymap.to_table())
# print("all sky integral is: ", full_skymap.allsky_integral())
# print(full_skymap.nside, full_skymap.nside2ang(), full_skymap.pixels)
# nu = full_skymap.neutrinoskymap(31.5, -41.43, 0.5)
# nu_skymap = HealPixSkymap(nu.s, uniq=nu.u).rasterize(pad=0, as_skymap=True)

# print(nu_skymap.to_table())

from coincidence_sig import TS
from data_loading import retrieve_event

# cwb_skymap = retrieve_event("S250201al")[0]
# print(cwb_skymap)

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
    gw_skymap_path = Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4a/S230627c_Bilby.multiorder.fits") # 1e-26
    skymap = HealPixSkymap.load_locally(gw_skymap_path, burst=False).rasterize(as_skymap=True)
    graceid = gw_skymap_path.name.split('_')[0]
    far_path = false_alarm_rate_path / (graceid + "_far.npy")
    far = np.load(far_path)
    neutrino_list = [IceCubeNeutrino(time_mjd, 160.1, 48.1, 1.2, 10**4.5)]
    skymap.plot(neutrino_list)
    plt.show()
    print(TS(time_gps, skymap, far, neutrino_list))

# gw_event()

import glob
import os

# Set the directory containing your .npy files
directory = '/home/aki/snakepit/multi_messenger_astro/core/nullstat'

# Get list of all .npy files
file_list = sorted(glob.glob(os.path.join(directory, '*.npy')))

# Load and concatenate
all_null_stats = np.concatenate([np.load(f) for f in file_list])
all_null_stats = all_null_stats[all_null_stats <= 1.0]
all_null_stats = all_null_stats[all_null_stats != 0.0]

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

plt.figure(figsize=(8, 5))
plt.hist(log_null_stats, bins=100, color='skyblue', edgecolor='black')
plt.xlabel(r'$-\log_{10}$(test statistic)')
plt.ylabel('Frequency')
plt.title('Distribution of Null Test Statistics')
plt.grid(True)
plt.show()
