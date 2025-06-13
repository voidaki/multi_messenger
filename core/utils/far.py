"""
author:voidaki
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time

from data_loading import load_gravitational_wave_data

gw_data = load_gravitational_wave_data()

location = EarthLocation(lon=0 * u.deg, lat=0 * u.deg, height=0 * u.m)


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
    while gw_data["far_gstlal"][index] >= 2 or gw_data["far_mbta"][index] >= 2 or gw_data["far_pycbc_hyperbank"] >= 2:
        count += 1
        if count == len(gw_data["gpstime"]):
            return np.inf
        else:
            index = deviation_sorted[count]
    return np.min(
        [
            (gw_data["far_gstlal"][index] <= 2) * gw_data["far_gstlal"][index],
            (gw_data["far_mbta"][index] <= 2) * gw_data["far_mbta"][index],
            (gw_data["far_pycbc_hyperbank"][index] <= 2) * gw_data["far_pycbc_hyperbank"][index]
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

    cos_angular_distance = np.sin(gw_data["altitude"]) * np.sin(alt) + np.cos(
        gw_data["altitude"]
    ) * np.cos(alt) * np.cos(gw_data["azimuth"] - az)
    cos_angular_distance[cos_angular_distance > 1] = 1
    cos_angular_distance[cos_angular_distance < -1] = -1
    angular_distance = np.arccos(cos_angular_distance) * 180 / np.pi

    sky_location_condition = angular_distance <= 15.0
    distance_condition = (gw_data["distance"] - distance) ** 2 <= (distance / 5) ** 2
    mass_condition = (
        (mass1 * 0.65 <= gw_data["mass1"])
        & (gw_data["mass1"] <= mass1 * 1.35)
        & (mass2 * 0.65 <= gw_data["mass2"])
        & (gw_data["mass2"] <= mass2 * 1.35)
    )

    total_condition = sky_location_condition & distance_condition & mass_condition

    pre_far_list = np.concatenate(
        (
            gw_data["far_gstlal"][total_condition],
            gw_data["far_mbta"][total_condition],
            gw_data["far_pycbc_hyperbank"][total_condition],
        )
    )
    far_list = pre_far_list[pre_far_list <= 2]

    return far_list
