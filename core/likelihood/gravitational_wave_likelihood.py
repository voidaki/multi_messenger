import numpy as np
from scipy.stats import norm
import astropy.units as u
import astropy_healpix as ah
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from tqdm import tqdm

from data_loading import load_gravitational_wave_data


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

location = EarthLocation(lon=0 * u.deg, lat=0 * u.deg, height=0 * u.m)


def volume_localization(skymap, distance, right_asc, declination):
    """Yields the probability of the 3 dimensional point in space: distance and sky location"""
    uniq = skymap["UNIQ"]
    prob_dens = skymap["PROBDENSITY"]
    distmu = skymap["DISTMU"]
    distsigma = skymap["DISTSIGMA"]
    distnorm = skymap["DISTNORM"]

    level, ipix = ah.uniq_to_level_ipix(uniq)
    nside = ah.level_to_nside(level)  # Lateralization (from uniq to nested)

    ra, dec = right_asc * u.deg, declination * u.deg  # Float into degrees

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


def P_signal_GW_t(t_GW, t_s):  # Signal likelihood temporal distribution function
    t_plus = 250
    t_minus = -250
    if t_minus <= t_GW - t_s <= t_plus:
        return (t_plus - t_minus) ** -1
    else:
        return 0


def cut_off(false_alarm_rate):
    """Cut off function of the false alarm, eliminates the low-significance events"""
    if false_alarm_rate <= 2:  # 2 per day
        return 1
    else:
        return 0


def match_far(gpstime, distance, right_ascension, declination, mass1, mass2):
    """Finds a false alarm rate from the O3 Sensitivity measurements data set for the given parameters"""
    t = Time(gpstime * u.s, format="gps")
    altaz = AltAz(obstime=t, location=location)
    observation = SkyCoord(
        ra=right_ascension * u.deg, dec=declination * u.deg, frame="icrs"
    )
    observation_alt_az = observation.transform_to(altaz)
    alt = observation_alt_az.alt.value
    az = observation_alt_az.az.value

    cos_angular_distance = np.sin(gw_data["altitude"]) * np.sin(alt) + np.cos(
        gw_data["altitude"]
    ) * np.cos(alt) * np.cos(gw_data["azimuth"] - az)
    cos_angular_distance[cos_angular_distance > 1] = 1
    cos_angular_distance[cos_angular_distance < -1] = -1
    angular_distance = np.arccos(cos_angular_distance) * 180 / np.pi

    deviation_sky_location = angular_distance**2 / ((5) ** 2)
    # deviation_inc = (np.cos(inc_source) - np.cos(inclination))**2/0.1**2
    deviation_distance = (gw_data["distance"] - distance) ** 2 / (distance / 10) ** 2
    deviation_mass = ((gw_data["mass1"] - mass1) + (gw_data["mass2"] - mass2)) ** 2 / (
        (mass1 + mass2) * 0.2
    ) ** 2

    total_deviation = deviation_sky_location + deviation_distance + deviation_mass

    count = 0
    deviation_sorted = np.argsort(total_deviation)
    index = deviation_sorted[count]
    while gw_data["far_gstlal"][index] >= 2 or gw_data["far_mbta"][index] >= 2:
        count += 1
        if count == len(gw_data["gpstime"]):
            return np.inf
        else:
            index = deviation_sorted[count]
    return np.min(
        [
            (gw_data["far_gstlal"][index] <= 2) * gw_data["far_gstlal"][index],
            (gw_data["far_mbta"][index] <= 2) * gw_data["far_mbta"][index],
        ]
    )


def far_list(gpstime, distance, right_ascension, declination, mass1, mass2):
    """Returns a list of false alarms that are match the parameters to a certain binwidth for each."""
    t = Time(gpstime * u.s, format="gps")
    altaz = AltAz(obstime=t, location=location)
    observation = SkyCoord(
        ra=right_ascension * u.deg, dec=declination * u.deg, frame="icrs"
    )
    observation_alt_az = observation.transform_to(altaz)
    alt = observation_alt_az.alt.value
    az = observation_alt_az.az.value

    cos_angular_distance = np.sin(alt_source) * np.sin(alt) + np.cos(
        alt_source
    ) * np.cos(alt) * np.cos(az_source - az)
    cos_angular_distance[cos_angular_distance > 1] = 1
    cos_angular_distance[cos_angular_distance < -1] = -1
    angular_distance = np.arccos(cos_angular_distance) * 180 / np.pi

    sky_location_condition = angular_distance <= 15.0
    distance_condition = (distance_source - distance) ** 2 <= (distance / 5) ** 2
    mass_condition = (
        (mass1 * 0.65 <= mass1_source)
        & (mass1_source <= mass1 * 1.35)
        & (mass2 * 0.65 <= mass2_source)
        & (mass2_source <= mass2 * 1.35)
    )

    total_condition = sky_location_condition & distance_condition & mass_condition

    pre_far_list = np.concatenate(
        (
            far_gstlal[total_condition],
            far_mbta[total_condition],
            far_pycbc_hyperbank[total_condition],
        )
    )
    far_list = pre_far_list[pre_far_list <= 2]

    return far_list


def P_F(
    false_alarm_rate, gpstime, distance, right_ascension, declination, mass1, mass2
):
    fars = far_list(gpstime, distance, right_ascension, declination, mass1, mass2)
    if len(fars) == 0:
        return 0
    else:
        deviation_far = (false_alarm_rate - fars) ** 2 / (false_alarm_rate / 10) ** 2
        count = 0
        for deviation in deviation_far:
            if deviation <= 5:
                count += 1
        return count / len(fars)


def P_signal_GW_source(gpstime, distance, ra, dec, M_1, M_2):
    """Retuns the source probability distribution for the gravitational wave source."""
    far = match_far(gpstime, distance, ra, dec, M_1, M_2)
    r_max = max(distance_source)
    normalization_r = r_max**3 / 3  # normalizing the distance distribution
    m1_max, m1_min = np.max(mass1_source), np.min(mass1_source)
    m2_max, m2_min = np.max(mass2_source), np.min(mass2_source)
    normalization_mass1, normalization_mass2 = (
        np.log(m1_max / m1_min),
        np.log(m2_max / m2_min),
    )  # normalization of the mass distributions
    nominator = distance**2 / (4 * np.pi) * cut_off(far)
    denominator = (
        M_1 * M_2 * normalization_r * normalization_mass1 * normalization_mass2
    )
    return nominator / denominator


def P_signal_GW_F(false_alarm_rate):
    """Monte-Carlo integration over all parameters for gravitational waves
    for finding the probability of any given false alarm rate."""
    far_mask = np.logical_or(far_gstlal <= 2, far_mbta <= 2)
    gpstime_filtered = gpstime_source[far_mask]
    distance_filtered = distance_source[far_mask]
    ra_filtered = ra_source[far_mask]
    dec_filtered = dec_source[far_mask]
    mass1_filtered = mass1_source[far_mask]
    mass2_filtered = mass2_source[far_mask]

    N = len(gpstime_filtered)
    n_steps = N // 25
    integral_sum = 0

    # Add progress bar
    for i in tqdm(range(n_steps), desc="Computing integral"):
        gpstime = gpstime_filtered[i]
        r_s = distance_filtered[i]
        ra = ra_filtered[i]
        dec = dec_filtered[i]
        M1 = mass1_filtered[i]
        M2 = mass2_filtered[i]

        p_far = P_F(false_alarm_rate, gpstime, r_s, ra, dec, M1, M2)
        if p_far == 0:
            p_source = 0
        else:
            p_source = P_signal_GW_source(gpstime, r_s, ra, dec, M1, M2)

        integral_sum += p_far * p_source

    integral_value = integral_sum / n_steps
    return integral_value


def P_far(false_alarm_rate):
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


def P_null_GW_F(significance):
    if significance == "low-significance":
        return 2.3 * 10**-5  # Hz or 2 per day
    if significance == "significant":
        return 6.34 * 10**-8  # Hz or once per six months
