"""
author:voidaki
"""
import numpy as np
from scipy.stats import norm
import astropy.units as u
import astropy_healpix as ah
from tqdm import tqdm

from data_loading import load_gravitational_wave_data
from utils.far import far_list

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


def volume_localization(skymap, distance, right_ascension, declination):
    """Yields the probability of the 3 dimensional point in space: distance and sky location"""
    uniq = skymap["UNIQ"]
    prob_dens = skymap["PROBDENSITY"]
    distmu = skymap["DISTMU"]
    distsigma = skymap["DISTSIGMA"]
    distnorm = skymap["DISTNORM"]

    level, ipix = ah.uniq_to_level_ipix(uniq)
    nside = ah.level_to_nside(level)  # Lateralization (from uniq to nested)

    ra, dec = right_ascension * u.deg, declination * u.deg  # Float into degrees

    match_ipix = ah.lonlat_to_healpix(
        ra, dec, nside, order="nested"
    )  # longtitude and latitude into
    i = np.flatnonzero(ipix == match_ipix)[0]
    prob_per_deg2 = prob_dens[i].to_value(
        u.deg**-2
    )  # Total probability of that sky location per degrees^2
    r = distance * u.Mpc
    prob_r = (
        r**2 * distnorm[i] * norm(distmu[i], distsigma[i]).pdf(r)
    )  # Probability (Gaussian) distribution of distance in the given sky location

    prob = prob_per_deg2 * prob_r

    return prob


def temporal_distribution(t_GW, t_s):  # Signal likelihood temporal distribution function
    t_plus = 250
    t_minus = -250
    if t_minus <= t_GW - t_s <= t_plus:
        return (t_plus - t_minus) ** -1
    else:
        return 0


def Pfar(false_alarm_rate):
    """Returns the probability of getting any given false alarm rate using
    the O3 Sensitivity Measurements dataset for the binwidths. Only including
    the false alarm rates below 2 per day since we only get measurements of them."""
    far_gstlal_filtered = far_gstlal[far_gstlal <= 2]
    far_mbta_filtered = far_mbta[far_mbta <= 2]
    far_pycbc_hyperbank_filtered = far_pycbc_hyperbank[far_pycbc_hyperbank <= 2]
    far_total = np.concatenate(
        (far_gstlal_filtered, far_mbta_filtered, far_pycbc_hyperbank_filtered)
    )  # All false alarm rates that are less than 2 per day
    deviation_far = (false_alarm_rate - far_total) ** 2 <= (false_alarm_rate / 6.5) ** 2
    prob = np.sum(deviation_far) / (len(far_total))
    return prob


def P_null_GW_t(t_GW, t_start, t_end):
    if t_start <= t_GW <= t_end:
        T_obs = t_end - t_start
        return T_obs**-1
    else:
        return 0


