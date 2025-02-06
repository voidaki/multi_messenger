import numpy as np

from data_loading import load_gravitational_wave_data
from utils.far import match_far, cut_off

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

def temporal(t_s, T_obs):
    return T_obs**-1

def distance(r_s):
    r_max = max(distance_source)
    normalization_r = r_max**3 / 3  # normalizing the distance distribution
    return r_s**2 / normalization_r

def angular(right_ascension, declination):
    return (4*np.pi)**-1

def mass(mass1, mass2):
    m1_max, m1_min = np.max(mass1_source), np.min(mass1_source)
    m2_max, m2_min = np.max(mass2_source), np.min(mass2_source)
 
    normalization_mass1, normalization_mass2 = (
        np.log(m1_max / m1_min),
        np.log(m2_max / m2_min),
    )  # normalization of the mass distributions
    return mass1**-1 * mass2**-1 * normalization_mass1**-1 * normalization_mass2**-1
 
def neutrino_energy(total_E):
    normalization_total_E = np.log(10**5)
    return total_E**-1 * normalization_total_E**-1

def gravitational_conditional(gpstime, distance, ra, dec, M_1, M_2):
    """Retuns the source probability distribution for the gravitational wave source."""
    far = match_far(gpstime, distance, ra, dec, M_1, M_2)
    T_obs = 500
    return temporal(gpstime, T_obs)*distance(distance)*angular(ra, dec)*mass(M_1, M_2)*cut_off(far)
