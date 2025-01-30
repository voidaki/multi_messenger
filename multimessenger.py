import numpy as np
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt
from ligo.gracedb.rest import GraceDb
import astropy.units as u
from astropy.table import QTable
from astropy.utils.data import download_file
import astropy_healpix as ah
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import h5py
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm

client = GraceDb()

"""Gravitational wave likelihood section is below."""

def retrieve_event(event_name):
    """Retreaving the skymap from GraceDB and reading in Qtable format"""
    gw = client.superevent(event_name)
    gw_dict = gw.json()
    t_GW = gw_dict.get("t_0")
    far = gw_dict.get("far")
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


# Getting Volume Localization
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


injection_file = "data/gw_data/O3_sensitivity/endo3_bnspop-LIGO-T2100113-v12.hdf5"
# injection_file = "endo3_bbhpop-LIGO-T2100113-v12.hdf5"

with h5py.File(injection_file, "r") as f:
    N_draw = f.attrs["total_generated"]

    gpstime_source = f["injections/gps_time"][:]  # s
    mass1_source = f["injections/mass1_source"][:]  # M_solar
    mass2_source = f["injections/mass2_source"][:]  # M_solar
    distance_source = f["injections/distance"][:]  # Mpc
    inc_source = (f["injections/inclination"][:] * 180 / np.pi)  # radians to degrees (-90, 90)
    dec_source = (f["injections/declination"][:] * 180 / np.pi)  # radians to degrees (-90, 90)
    ra_source = (f["injections/right_ascension"][:] * 180 / np.pi)  # radians to degrees (0,360)
    # far_cwb = f['injections/far_cwb'][:] # Off for the bns systems
    far_gstlal = f["injections/far_gstlal"][:] / 365.25  # per day
    far_mbta = f["injections/far_mbta"][:] / 365.25  # per day
    far_pycbc_hyperbank = f["injections/far_pycbc_hyperbank"][:] / 365.25  # per day
    far_pycbc_bbh = f["injections/far_pycbc_bbh"][:] / 365.25  # per day


gpstime_source = np.array(gpstime_source)
mass1_source = np.array(mass1_source)
mass2_source = np.array(mass2_source)
distance_source = np.array(distance_source)
inc_source = np.array(inc_source)
ra_source = np.array(ra_source)
dec_source = np.array(dec_source)
far_gstlal = np.array(far_gstlal)
far_mbta = np.array(far_mbta)
far_pycbc_hyperbank = np.array(far_pycbc_hyperbank)
far_pycbc_bbh = np.array(far_pycbc_bbh)

alt_source = np.concatenate(
    (
        np.load("data/gw_data/alt_array0.npy"),
        np.load("data/gw_data/alt_array1.npy"),
        np.load("data/gw_data/alt_array2.npy"),
        np.load("data/gw_data/alt_array3.npy"),
    )
)
az_source = np.concatenate(
    (
        np.load("data/gw_data/az_array0.npy"),
        np.load("data/gw_data/az_array1.npy"),
        np.load("data/gw_data/az_array2.npy"),
        np.load("data/gw_data/az_array3.npy"),
    )
)

location = EarthLocation(lon=0 * u.deg, lat=0 * u.deg, height=0 * u.m)


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

    cos_angular_distance = np.sin(alt_source) * np.sin(alt) + np.cos(
        alt_source
    ) * np.cos(alt) * np.cos(az_source - az)
    cos_angular_distance[cos_angular_distance > 1] = 1
    cos_angular_distance[cos_angular_distance < -1] = -1
    angular_distance = np.arccos(cos_angular_distance) * 180 / np.pi

    deviation_sky_location = angular_distance**2 / ((5) ** 2)
    # deviation_inc = (np.cos(inc_source) - np.cos(inclination))**2/0.1**2
    deviation_distance = (distance_source - distance) ** 2 / (distance / 10) ** 2
    deviation_mass = ((mass1_source - mass1) + (mass2_source - mass2)) ** 2 / (
        (mass1 + mass2) * 0.2
    ) ** 2

    total_deviation = deviation_sky_location + deviation_distance + deviation_mass

    count = 0
    deviation_sorted = np.argsort(total_deviation)
    index = deviation_sorted[count]
    while far_gstlal[index] >= 2 or far_mbta[index] >= 2:
        count += 1
        if count == len(gpstime_source):
            return np.inf
        else:
            index = deviation_sorted[count]
    return np.min(
        [
            (far_gstlal[index] <= 2) * far_gstlal[index],
            (far_mbta[index] <= 2) * far_mbta[index],
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

"""Neutrino Likelihood Section is below."""

effectiveArea_path = Path("./data/neutrino_data/irfs")
events_path = Path("./data/neutrino_data/events")
dataframes_effectiveArea = {}
dataframes_events = {}

# Separating the data files column names into their proper place
for file_path in effectiveArea_path.glob("*effectiveArea.csv"):
    with open(file_path, "r") as file:
        first_row = file.readline().strip()

    column_names = re.split(r"\s{2,}", first_row)
    column_names = [name for name in column_names if name != "#"]
    df = pd.read_csv(file_path, sep=r"\s+", skiprows=1, header=None)
    df.columns = column_names
    key = file_path.stem
    dataframes_effectiveArea[key] = df

for file_path in events_path.glob("*.csv"):
    with open(file_path, "r") as file:
        first_row = file.readline().strip()

    column_names = re.split(r"\s{2,}", first_row)
    column_names = [name for name in column_names if name != "#"]
    df = pd.read_csv(file_path, sep=r"\s+", skiprows=1, header=None)
    df.columns = column_names
    key = file_path.stem
    dataframes_events[key] = df


def P_signal_nu_t(t_nu, t_s):
    t_plus = 250
    t_minus = -250
    if t_minus <= t_nu - t_s <= t_plus:
        return (t_plus - t_minus) ** -1
    else:
        return 0


def P_source_loc_detection(
    right_ascension_nu,
    declination_nu,
    angular_error,
    right_ascension_source,
    declination_source,
):
    """Normal distribution of the sky location and the angular error."""
    angle_difference = (right_ascension_nu - right_ascension_source) ** 2 + (
        declination_nu - declination_source
    ) ** 2
    return np.exp(-angle_difference / (2 * angular_error**2)) / (
        2 * np.pi * angular_error**2
    )


def A_eff(energy_log10, declination):
    #     energy_log10 = np.log10(energy)
    df = dataframes_effectiveArea["IC86_II_effectiveArea"]
    condition = (
        (df["log10(E_nu/GeV)_min"] <= energy_log10)
        & (energy_log10 < df["log10(E_nu/GeV)_max"])
        & (df["Dec_nu_min[deg]"] <= declination)
        & (declination < df["Dec_nu_max[deg]"])
    )
    effective_area = df[condition]
    return effective_area.iloc[0]["A_Eff[cm^2]"]


def P_signal_nu_source(declination_source):
    """Monte-Carlo integration of the source parameter distribution
    of sky location for the high energy neutrinos."""
    df = dataframes_effectiveArea["IC86_II_effectiveArea"]
    condition = (df["Dec_nu_min[deg]"] <= declination_source) & (
        declination_source < df["Dec_nu_max[deg]"]
    )
    filtered_df = df[condition]
    filtered_df = filtered_df[filtered_df.columns[:]].to_numpy()
    areas_top = []
    areas_bottom = []
    for row in filtered_df:
        d_epsilon = 0.2
        epsilon = (row[1] - row[0]) / 2
        d_dec = row[3] - row[2]
        A_eff = row[4]
        area = d_epsilon * d_dec * A_eff * epsilon**-2  # Area under the rectangle
        areas_top.append(area)
    integral_top = np.sum(areas_top)

    df = df[df.columns[:]].to_numpy()
    for row in df:
        d_epsilon = 0.2
        epsilon = (row[1] - row[0]) / 2
        d_dec = row[3] - row[2]
        A_eff = row[4]
        area = d_epsilon * d_dec * A_eff * epsilon**-2  # Area under the rectangle
        areas_bottom.append(area)
    integral_bottom = np.sum(areas_bottom)

    return integral_top / integral_bottom


def P_signal_nu_epsilon(epsilon_nu):
    """Monte-Carlo integration of the energy distributions for
    the high energy neutrinos. Returns the probability of getting
    a certain energy."""
    df = dataframes_effectiveArea["IC86_II_effectiveArea"]
    condition = (df["log10(E_nu/GeV)_min"] <= epsilon_nu) & (
        epsilon_nu < df["log10(E_nu/GeV)_max"]
    )
    filtered_df = df[condition]
    filtered_df = filtered_df[filtered_df.columns[:]].to_numpy()
    areas_top = []
    areas_bottom = []
    for row in filtered_df:
        d_epsilon = 0.2
        d_dec = row[3] - row[2]
        A_eff = row[4]
        P_source = P_signal_nu_source(row[2] + 1)
        area = (
            d_epsilon * d_dec * A_eff * P_source * epsilon_nu**-2
        )  # Area under the rectangle
        areas_top.append(area)
    integral_top = np.sum(areas_top)

    df = df[df.columns[:]].to_numpy()
    for row in df:
        d_epsilon = 0.2
        epsilon = (row[1] - row[0]) / 2
        d_dec = row[3] - row[2]
        A_eff = row[4]
        P_source = P_signal_nu_source(row[2] + 1)
        area = (
            d_epsilon * d_dec * A_eff * P_source * epsilon**-2
        )  # Area under the rectangle
        areas_bottom.append(area)
    integral_bottom = np.sum(areas_bottom)

    return integral_top / integral_bottom


def P_sigma_nu_E(epsilon_nu):
    """Epsilon nu is in the units of log(E_nu/GeV)"""
    df = dataframes_events["IC86_VII_exp-1"]
    binwidth = 0.01
    lower_bound = epsilon_nu - binwidth
    upper_bound = epsilon_nu + binwidth

    bin_data = [
        epsilon
        for epsilon in df["log10(E/GeV)"]
        if lower_bound <= epsilon <= upper_bound
    ]

    probability = len(bin_data) / len(df["log10(E/GeV)"])
    return probability


def P_skyloc_source(right_ascension, declination):
    """Just checks the angles if they are in the spherical sky map ranges
    and returns a uniform distribution over the entire sky."""
    if (0 <= right_ascension <= 360) & (-90 <= declination <= 90):
        return (4*np.pi)**-1
    return 0


def P_skyloc_data(declination, epsilon_nu):
    """Returns the probability of the given sky location and energy level
    in log10 scale from the data so can be used for the signal hypothesis."""
    df = dataframes_events["IC86_VII_exp-1"]
    condition_epsilon = (df["log10(E/GeV)"] <= epsilon_nu + 0.01) & (
        epsilon_nu - 0.01 <= df["log10(E/GeV)"]
    )
    condition_dec = (df["Dec[deg]"] <= declination + 1) & (
        declination - 1 <= df["Dec[deg]"]
    )
    condition = condition_epsilon & condition_dec
    filtered_df = df[condition]
    return len(filtered_df) if len(filtered_df) != 0 else 0


def P_skyloc_Aeff(declination, epsilon_nu):
    """Returns the probability of the given sky location and energy level
    using the effective area."""
    df = dataframes_effectiveArea["IC86_II_effectiveArea"]
    condition_epsilon = (df["log10(E_nu/GeV)_min"] <= epsilon_nu) & (
        epsilon_nu <= df["log10(E_nu/GeV)_max"]
    )
    condition_declination = (df["Dec_nu_min[deg]"] <= declination) & (
        declination <= df["Dec_nu_max[deg]"]
    )
    condition = condition_epsilon & condition_declination
    filtered_df = df[condition]
    filtered_df = filtered_df[filtered_df.columns[:]].to_numpy() 
    effective_area = filtered_df[0][4]
    return effective_area * (1/epsilon_nu**2) * (4*np.pi)**-1


if __name__ == "__main__":
    P_Aeffvec = np.vectorize(P_skyloc_Aeff)
    P_datavec = np.vectorize(P_skyloc_data)
    
    epsilon = 5
    declination_vals = np.linspace(-89, 89, 500)
    p_emprical = P_datavec(declination_vals, epsilon)
    p_Aeff = P_Aeffvec(declination_vals, epsilon)
    plt.plot(declination_vals, p_Aeff, label=f"Effective Area, epsilon = {epsilon}")
    plt.plot(declination_vals, p_emprical, label=f"Empirical Data, epsilon = {epsilon}")
    plt.legend()
    plt.show()
    #skymap, gpstime, false_alarm_rate = retrieve_event("S190425z")
    #distance = 159
    #right_ascension = 241.69921874999997
    #declination = 22.993394314297802
    #M_1 = 1.74
    #M_1 = 1.56
