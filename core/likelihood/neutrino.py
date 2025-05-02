import numpy as np
import pandas as pd

from data_loading import load_neutrino_data
from utils import (
    Aeff,
    search_parameters
    )

neutrino_data = load_neutrino_data()
dataframes_effectiveArea = neutrino_data["effective_areas"]
dataframes_events = neutrino_data["events"]


def temporal_distribution(t_nu, t_s):
    t_plus = 250
    t_minus = -250
    if t_minus <= t_nu - t_s <= t_plus:
        return (t_plus - t_minus) ** -1
    else:
        return 0


def source_localization(right_ascension_nu, declination_nu, angular_error, right_ascension_source, declination_source):
    """Normal distribution of the sky location and the angular error."""
    angle_difference_squared = (right_ascension_nu - right_ascension_source)**2 + (
        declination_nu - declination_source)**2
    return np.exp(-angle_difference_squared / (2 * angular_error**2)) / (2*np.pi*angular_error**2)


def uniform_allsky(right_ascension, declination):
    """Just checks the angles if they are in the spherical sky map ranges
    and returns a uniform distribution over the entire sky."""
    if (0 <= right_ascension <= 360) and (-90 <= declination <= 90):
        return (4*np.pi)**-1
    return 0


# def Paeffe(epsilon, declination, search_params=search_parameters("bns")):
#     """Returns the probability of the given sky location and energy level
#     using the effective area.
    
#     Parameters
#     ----------
#     epsilon: float
#         Reconstructed energy of the neutrino, in GeV or log10(E/GeV).
#     declination: float
#         Declination angle of the IceCube neutrino, in degrees
#     """
#     if epsilon >= 100.0:
#         epsilon = np.log10(epsilon)
#     # Clipping neutrino energy inside the bounds
#     if epsilon < np.log10(search_params.epsilonmin):
#         epsilon = np.log10(search_params.epsilonmin)
#     if epsilon > np.log10(search_params.epsilonmax):
#         epsilon = np.log10(search_params.epsilonmax)
    
#     df = neutrino_data["effective_areas"]["IC86_II_effectiveArea"]
#     condition_epsilon = (df['log10(E_nu/GeV)_min'] <= epsilon) & (epsilon <= df['log10(E_nu/GeV)_max'])
#     condition_declination = (df['Dec_nu_min[deg]'] <= declination) & (declination <= df['Dec_nu_max[deg]'])
#     condition = condition_epsilon & condition_declination
#     filtered_df = df[condition]
#     filtered_df = filtered_df[filtered_df.columns[:]].to_numpy()
    
#     A_eff = filtered_df[0][4]
   
#     return A_eff*(10**epsilon)**-2*(4*np.pi)**-1

def Paeffe(epsilon, declination, search_params=search_parameters("bns")):
    A_eff = Aeff(epsilon, declination, search_params)
    return A_eff*epsilon**-2*(4*np.pi)**-1


def Pempe(epsilon, declination, search_params=search_parameters("bns")):
    """Returns the probability of the given sky location and energy level
    in log10 scale from the data so can be used for the signal hypothesis.
    
    Parameters
    ----------
    epsilon: float
        Reconstructed energy of the neutrino, in GeV or log10(E/GeV).
    declination: float
        Declination angle of the IceCube neutrino, in degrees
    """
    import pandas as pd
    
    # Converting to log10 scale if energy was entered in GeVs
    if epsilon >= 100.0:
        epsilon = np.log10(epsilon)
    # Clipping neutrino energy inside the bounds
    if epsilon < np.log10(search_params.epsilonmin):
        epsilon = np.log10(search_params.epsilonmin)
    if epsilon > np.log10(search_params.epsilonmax):
        epsilon = np.log10(search_params.epsilonmax)

    dataframes=dataframes_events
    epsilonbins = np.linspace(np.log10(search_params.epsilonmin), np.log10(search_params.epsilonmax), 50)
    decbins = np.linspace(-90.0, 90.0, 100)

    emp_epsilon = np.array(pd.concat([dataframes[df]["log10(E/GeV)"] for df in dataframes], axis=0))
    emp_dec = np.array(pd.concat([dataframes[df]["Dec[deg]"] for df in dataframes], axis=0))

    hist, x_edges, y_edges = np.histogram2d(emp_epsilon, emp_dec, bins=[epsilonbins, decbins], density=True)

    epsbin_i = np.digitize(epsilon, x_edges) - 1  
    decbin_i = np.digitize(declination, y_edges) - 1  
    
    # Check if the values fall within the bin range
    if 0 <= epsbin_i < len(x_edges) - 1 and 0 <= decbin_i < len(y_edges) - 1:
        return hist[epsbin_i, decbin_i]
    else:
        return 0


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
