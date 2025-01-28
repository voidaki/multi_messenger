import math
import numpy as np
import json
from scipy.stats import  norm, uniform
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from ligo.gracedb.rest import GraceDb
client = GraceDb()
from astropy.io import fits
import astropy.units as u
from astropy.table import QTable
from astropy.cosmology import FlatLambdaCDM, z_at_value, Planck13
from astropy.utils.data import download_file
import astropy_healpix as ah
from astropy.coordinates import ICRS, Galactic
import h5py
import statistics
import pandas as pd


def ninty_perc(skymap): # Finding the 90% Credible Prob. Region
    skymap.sort("PROBDENSITY", reverse=True)
    level, ipix = ah.uniq_to_level_ipix(skymap["UNIQ"])
    nside = ah.level_to_nside(level)
    pixel_area = ah.nside_to_pixel_area(nside)

    prob = pixel_area*skymap["PROBDENSITY"]
    cumprob = np.cumsum(prob)
    i = cumprob.searchsorted(0.9)

    area_90 = pixel_area[:i].sum()
    area_90_d2 = area_90.to_value(u.deg**2)
    
    print(area_90, "and ", area_90_d2, "per degrees^2")
    skymap = skymap[:i]
    skymap.sort("UNIQ")
    skymap = skymap["UNIQ",]
    # skymap.write("90percent.moc.fits")
    return skymap


def retrieve_event(event_name): # Retreaving the skymap from GraceDB and reading in Qtable format
    gw = client.superevent(event_name)
    gw_dict = gw.json()
    t_GW = gw_dict.get('t_0')
    far = gw_dict.get('far')
    files = client.files(event_name).json()
    if "bayestar.multiorder.fits" in files:
        skymap_url = files["bayestar.multiorder.fits"]
    elif "Bilby.multiorder.fits" in files:
        skymap_url = files["Bilby.multiorder.fits"]
    else:
        multiorder_maps = [s for s in files if s.endswith("multiorder.fits")]
        skymap_url = files[multiorder_maps[0]]
    filename = download_file(skymap_url, cache=True)
    
    skymap = QTable.read(filename)
    return skymap, t_GW, far


def volume_localization(skymap, distance, right_asc, declination): # Yields the probability of the 3 dimensional point in space: distance and sky location
    uniq = skymap["UNIQ"]
    prob_dens = skymap["PROBDENSITY"]
    distmu = skymap["DISTMU"]
    distsigma = skymap["DISTSIGMA"]
    distnorm = skymap["DISTNORM"]
    
    level, ipix = ah.uniq_to_level_ipix(uniq)
    nside = ah.level_to_nside(level) # Lateralization (from uniq to nested)
    
    ra, dec = right_asc*u.deg, declination*u.deg # Float into degrees
    
    match_ipix = ah.lonlat_to_healpix(ra, dec, nside, order="nested") # longtitude and latitude into
    i = np.flatnonzero(ipix == match_ipix)[0]
    prob_per_deg2 = prob_dens[i].to_value(u.deg**-2) # Total probability of that sky location per degrees^2
    r = distance*u.Mpc
    prob_r = r**2*distnorm[i]*norm(distmu[i], distsigma[i]).pdf(r) # Probability (Gaussian) distribution of distance in the given sky location
    
    prob = prob_per_deg2*prob_r 

    return prob


injection_file = "endo3_bnspop-LIGO-T2100113-v12-1256655642-12905976.hdf5"

with h5py.File(injection_file, 'r') as f:
    N_draw = f.attrs["total_generated"]
    
    mass1_source = f['injections/mass1_source'][:] # M_solar
    mass2_source = f['injections/mass2_source'][:] # M_solar
    distance_source = f['injections/distance'][:] # Mpc
    dec_source = f['injections/declination'][:]*180/np.pi # radians to degrees (-90, 90)
    ra_source = f['injections/right_ascension'][:]*180/np.pi # radians to degrees (0,360)
    # far_cwb = f['injections/far_cwb'][:] # Off for the bns systems
    far_gstlal = f['injections/far_gstlal'][:] # per day
    far_mbta = f['injections/far_mbta'][:] # per day
    far_pycbc_hyperbank = f['injections/far_pycbc_hyperbank'][:] # per day
    far_pycbc_bbh = f['injections/far_pycbc_bbh'][:]


mass1_source = np.array(mass1_source)
mass2_source = np.array(mass2_source)
distance_source = np.array(distance_source)
ra_source = np.array(ra_source)
dec_source = np.array(dec_source)
far_gstlal = np.array(far_gstlal)
far_mbta = np.array(far_mbta)
far_pycbc_hyperbank = np.array(far_pycbc_hyperbank)
far_pycbc_bbh = np.array(far_pycbc_bbh)


def P_signal_GW_t(t_GW, t_s): # Signal likelihood temporal distribution function
    t_plus = 250
    t_minus = -250
    if t_minus <= t_GW - t_s <= t_plus:
        return (t_plus - t_minus)**-1
    else:
        return 0


def cut_off(far): # Cut off function of the false alarm, eliminates the low-significance events
    if far <= 2: # 2 per day
        return 1
    else:
        return 0

    
def get_far(distance, right_ascension, declination, mass1, mass2): # Get false alarm rate for given parameters from O3 search sensitivity estimates
    mass_conditions = (mass1 - 0.3 <= mass1_source) & (mass1_source <= mass1 + 0.3) & \
                        (mass2 - 0.3 <= mass2_source) & (mass2_source <= mass2 + 0.3)
    distance_conditions = (distance - 60 <= distance_source) & (distance_source <= distance + 60)
    sky_location_condition = ((right_ascension - 5 <= ra_source) & (ra_source <= right_ascension + 5) &
                              (declination - 4 <= dec_source) & (dec_source <= declination + 4))
    
    total_condition = mass_conditions & distance_conditions & sky_location_condition
    
    far_list = far_gstlal[total_condition]
    
    if len(far_list) > 0:
        false_alarm_rate = np.min(far_list)
    else:
        false_alarm_rate = np.inf
    return false_alarm_rate, far_list

    
def P_signal_GW_source(distance, ra, dec, M_1, M_2): # source parameter distribution for gravitational wave
    far = get_far(distance, ra, dec, M_1, M_2)[0]
    r_max = max(distance_source)
    normalization_r = r_max**3/3 # normalizing the distance distribution
    m1_max, m1_min = max(mass1_source), min(mass1_source)
    m2_max, m2_min = max(mass2_source), min(mass2_source)
    normalization_mass1, normalization_mass2 = np.log(m1_max/m1_min), np.log(m2_max/m2_min) # normalization of the mass distributions
    nominator = distance**2/(4*np.pi)*cut_off(far)
    denominator = M_1*M_2*normalization_r*normalization_mass1*normalization_mass2
    return nominator/denominator


def P_F(false_alarm_rate, distance, right_ascension, declination, mass1, mass2): # P(F|source parameters)
    far_lower_bound, far_upper_bound = far_gstlal - far_gstlal/5, far_gstlal + far_gstlal/5 # FIXME: Find appropriate tolerances for the boundaries
    far_condition = (far_lower_bound <= false_alarm_rate) & (false_alarm_rate <= far_upper_bound)
    acceptable_fars = far_gstlal[far_condition]
    
    far_given_parameters = get_far(distance, right_ascension, declination, mass1, mass2)[1]
    if len(acceptable_fars) == 0:
        p_far = 0
    else:
        if not set(acceptable_fars).isdisjoint(far_given_parameters):
            p_far = len(far_given_parameters)**-1
        else:
            p_far = 0
    return p_far


def P_signal_GW_F(false_alarm_rate): # Monte Carlo Integration over the source parameters
    integral_sum = 0
    N = len(mass1_source)//25
    for i in range(N):
        r_s = distance_source[i]
        ra = ra_source[i]
        dec = dec_source[i]
        M1 = mass1_source[i]
        M2 = mass2_source[i]
        
        p_far = P_F(false_alarm_rate, r_s, ra, dec, M1, M2)
        if p_far == 0:
            p_source = 0
        else:
            p_source = P_signal_GW_source(r_s, ra, dec, M1, M2)
        
        # Multiply the likelihood and prior and add to the sum
        integral_sum += p_far * p_source
    
    # Average the sum to get the Monte Carlo estimate of the integral
    integral_value = integral_sum/N
    return integral_value

    
def P_signal_GW(t_GW, t_s, skymap, false_alarm_rate, distance, right_ascension, declination, M_1, M_2): # P(data|source_parameters, H_s)
    P_t = P_signal_GW_t(t_GW, t_s)
    volume_loc = volume_localization(skymap, distance, right_ascension, declination)
    P_far = P_signal_GW_F(false_alarm_rate)
    P_source = P_signal_GW_source(distance, right_ascension, declination, M_1, M_2)
    if P_source == 0:
        P_source = 1e-10
    return P_t*volume_loc*P_far*P_source**-1


def P_null_GW_t(t_GW, t_start, t_end):
    if t_start <= t_GW <= t_end:
        T_obs = t_end - t_start
        return T_obs**-1
    else:
        return 0
    
def P_null_GW_F(significance):
    if significance == 'low-significance':
        return 2.3*10**-5 # Hz or 2 per day
    if significance == 'significant':
        return 6.34*10**-8 # Hz or once per six months



skymap, t_GW, false_alarm_rate = retrieve_event("S190425z")
distance = 159
right_ascension = 241.69921874999997
declination = 22.993394314297802
M_1 = 1.74
M_2 = 1.56
# print(get_far(distance, right_ascension, declination, M_1, M_2))
# print(P_signal_GW_t(t_GW, t_GW))
# print(volume_localization(skymap, distance, right_ascension, declination))
# print(P_signal_GW_F(false_alarm_rate))
# print(P_signal_GW_source(distance, right_ascension, declination, M_1, M_2))
# print(P_signal_GW(t_GW, t_GW, skymap, far, distance, right_ascension, declination, M_1, M_2))


print(P_signal_GW(t_GW, t_GW, skymap, false_alarm_rate, distance, right_ascension, declination, M_1, M_2))
