import pandas as pd
import numpy as np

from data_loading import load_gravitational_wave_data, load_neutrino_data
from utils.far import match_far, cut_off
from pydantic import BaseModel, Field

GW_BG_FARS = np.loadtxt("/home/aki/snakepit/multi_messenger_astro/data/gw_data/false_alarm_list.csv", delimiter=",")  


def save_dict_to_hdf5(data_dict, filepath):
    """
    Save a dictionary of Pandas DataFrames to an HDF5 file.
    
    Args:
        data_dict (dict): Dictionary where keys are dataset names and values are DataFrames.
        filepath (str): Path to the output HDF5 file.
    """
    with pd.HDFStore(filepath, mode="w") as store:
        for key, df in data_dict.items():
            store.put(key, df)  # Save each DataFrame under its key
    print(f"Saved {len(data_dict)} DataFrames to {filepath}")


def load_hdf5_to_dict(filepath):
    """
    Load an HDF5 file into a dictionary of Pandas DataFrames.
    
    Args:
        filepath (str): Path to the HDF5 file.
    
    Returns:
        dict: Dictionary where keys are dataset names and values are DataFrames.
    """
    data_dict = {}
    with pd.HDFStore(filepath, mode="r") as store:
        for key in store.keys():
            data_dict[key.strip("/")] = store[key]  # Remove leading '/'
    return data_dict


def Poisson(k, lamb):
    """Poisson probability density function with lamb mean 
    and k observed events."""
    k = int(k)
    return lamb**k * np.exp(-lamb) / np.math.factorial(k)

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


def temporal(search_params):
    T_obs = search_params.tgwplus - search_params.tgwminus
    return T_obs**-1


def Pr(r, search_parameters):
    r_max = 700.0 # Mpc for BNS
    N_r = r_max**3 / 3  # normalizing the distance distribution
    return r**2 / N_r


def sky_dist():
    """Probability distribution of angular sky position of events. It is assumed be uniformly distributed across sky."""
    return (4*np.pi)**-1


def PMgw(mass1, mass2, search_params):
    """Probability of the source mass distribution of objects of merger that produced the gravitational wave."""
    m1_max, m1_min = search_params.Mgwmax, search_params.Mgwmin
    m2_max, m2_min = search_params.Mgwmax, search_params.Mgwmin
 
    normalization_mass1, normalization_mass2 = (
        np.log(m1_max / m1_min),
        np.log(m2_max / m2_min),
    )  # normalization of the mass distributions
    return mass1**-1 * mass2**-1 * normalization_mass1**-1 * normalization_mass2**-1
 

def PEnu(Enu, search_params):
    """Probability of isotropic equivalent neutrino emission energy Enu given in the signal hypothesis."""
    normalization_total_E = np.log(search_params.Enumax/search_params.Enumin)
    return Enu**-1 * normalization_total_E**-1


def gravitational_conditional(gpstime, distance, ra, dec, M_1, M_2):
    """Retuns the source probability distribution for the gravitational wave source."""
    far = match_far(gpstime, distance, ra, dec, M_1, M_2)
    T_obs = 500
    return temporal(T_obs)*distance(distance)*sky_dist(ra, dec)*PMgw(M_1, M_2)*cut_off(far)


def Pfar(far):
    """Returns the probability of getting any given false alarm rate using
    the O3 Sensitivity Measurements dataset for the binwidths. Only including
    the false alarm rates below 2 per day since we only get measurements of them.
    
    Parameters
    ----------
    far: False alarm rate (Hz)"""
    if far > 2.3e-5:
        far = 2.3e-5
    far_gstlal_filtered = far_gstlal[far_gstlal <= 2]
    far_mbta_filtered = far_mbta[far_mbta <= 2]
    far_pycbc_hyperbank_filtered = far_pycbc_hyperbank[far_pycbc_hyperbank <= 2]
    far_total = np.concatenate(
        (far_gstlal_filtered, far_mbta_filtered, far_pycbc_hyperbank_filtered)
    )/86400  # All false alarm rates that are less than 2 per day
    bins = np.logspace(-52, np.log10(2.3e-5), num=(52-5)+1)

    counts, bin_edges = np.histogram(far_total, bins=bins, density=True)

    bin_widths = np.diff(bin_edges)
    probabilities = counts * bin_widths

    if far < bin_edges[0] or far > bin_edges[-1]: # False alarm rate is out of range
        return 0
    
    bin_i = np.digitize(far, bin_edges) - 1

    return probabilities[bin_i] if 0 <= bin_i < len(probabilities) else 0 

def Pgw_bns_r(r):
    """
    Returns the prior probability distribution of distance using the O3 Sensitiity Measurements
    dataset. Only including the events with false alarm rate below 2 per day since LVK only 
    gets the measurements of them.
    """
    subthreshold_events = np.logical_or((far_gstlal <= 2), (far_mbta <= 2), (far_pycbc_hyperbank <= 2))
    
    distance_filtered = distance_source[subthreshold_events]
    
    counts, bin_edges = np.histogram(distance_filtered, bins=30, density=True)

    bin_widths = np.diff(bin_edges)
    probabilities = counts * bin_widths
    
    bin_i = np.digitize(r, bin_edges) - 1
    
    return probabilities[bin_i]

# def Aeff(epsilon, declination, dataframes_effectiveArea):
#     """Returns the corresponding effective area of the neurino detection with 
#     respect to the individual energy of detected neturino in log10 scale and 
#      declination angle. """
#     df = nu_data["effective_areas"]["IC86_II_effectiveArea"]
#     condition_epsilon = (df['log10(E_nu/GeV)_min'] <= epsilon) & (epsilon <= df['log10(E_nu/GeV)_max'])
#     condition_declination = (df['Dec_nu_min[deg]'] <= declination) & (declination <= df['Dec_nu_max[deg]'])
#     condition = condition_epsilon & condition_declination
#     filtered_df = df[condition]
#     filtered_df = filtered_df[filtered_df.columns[:]].to_numpy()
    
#     A_eff = filtered_df[0][4]
#     return A_eff

def angle_dict():
    return [-90.00, -73.74, -66.93, -61.64, -57.14, -53.13, -49.46, -46.05, -42.84,
            -39.79, -36.87, -34.06, -31.33, -28.69, -26.10, -23.58, -21.10, -18.66,
            -16.26, -13.89, -11.54, -9.21, -6.89, -4.59, -2.29,
            0.0, 2.29, 4.59, 6.89, 9.21, 11.54, 13.89, 16.26, 18.66, 21.10,
            23.58, 26.10, 28.69, 31.33, 34.06, 36.87, 39.79, 42.84, 46.05, 49.46,
            53.13, 57.14, 61.64, 66.93, 73.74, 90.00]

def epsilon_dict():
    return [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2,
            3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6,
            4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0,
            6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4,
            7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8,
            9.0, 9.2, 9.4, 9.6, 9.8, 10.0] # log10(epsilon/GeV)

# def Aeff(epsilon, declination, search_params):
#     """Returns the corresponding effective area of the neurino detection with 
#     respect to the individual energy of detected neturino in log10 scale and 
#     declination angle.
     
#     Parameters
#     ----------
#     epsilon: float
#         Reconstructed energy of the neutrino
#     declination: float
#         Declination angle of the neutrino observation
#     """
#     if epsilon >= 100.0:
#         epsilon = np.log10(epsilon)
#     # Clipping neutrino energy inside the bounds
#     epsilon = np.clip(epsilon, np.log10(search_params.epsilonmin), np.log10(search_params.epsilonmax))
    
#     declination = np.clip(declination, -90.0, 90.0)

#     df = nu_data["effective_areas"]["IC86_II_effectiveArea"]

#     dec_angles = np.array(angle_dict())
#     dec = abs(declination)
#     dec_index = np.sum(dec_angles < dec) - 1
#     epsilon_index = int((epsilon - 2.0)/0.2)
 
#     if declination < 0:
#         row = df[(df['log10(E_nu/GeV)_min'] == epsilon_dict()[epsilon_index]) & (df['Dec_nu_min[deg]'] == -dec_angles[dec_index])]
#     else:
#         row = df[(df['log10(E_nu/GeV)_min'] == epsilon_dict()[epsilon_index]) & (df['Dec_nu_min[deg]'] == dec_angles[dec_index])]

#     if not row.empty:
#         return row.iloc[0]['A_Eff[cm^2]']
#     else:
#         return 0.

def Aeff(epsilon, declination, search_params):
    """Returns the corresponding effective area of the neurino detection with 
    respect to the individual energy of detected neturino in log10 scale and 
    declination angle.
     
    Parameters
    ----------
    epsilon: float
        Reconstructed energy of the neutrino
    declination: float
        Declination angle of the neutrino observation
    """
    if epsilon >= 100.0:
        epsilon = np.log10(epsilon)

    # epsilon = np.clip(epsilon, np.log10(search_params.epsilonmin), np.log10(search_params.epsilonmax))
    
    declination = np.clip(declination, -90.0, 89.9)

    df = nu_data["effective_areas"]["IC86_II_effectiveArea"]

    epsilon_index = np.searchsorted(epsilon_dict(), epsilon, side='right') - 1
    dec_index = np.searchsorted(angle_dict(), declination, side='right') - 1
    

    row = df[(df['log10(E_nu/GeV)_min'] == np.array(epsilon_dict())[epsilon_index]) & (df['Dec_nu_min[deg]'] == np.array(angle_dict())[dec_index])]

    if not row.empty:
        return row.iloc[0]['A_Eff[cm^2]']
    else:
        # print(f"Encountered error, indexes are out of bounds! {epsilon_index}, {dec_index}",epsilon, declination)
        return 0.

def expnu_dec(dec, search_params):
    epsilon_bins = np.array(epsilon_dict()) 
    epsilon_vals = 10**((epsilon_bins[:-1] + epsilon_bins[1:]) / 2) 
    delta_eps = 10**epsilon_bins[1:] - 10**epsilon_bins[:-1]

    aeff_vals = np.array([Aeff(eps, dec, search_params) for eps in epsilon_vals])
    integrand_vals = aeff_vals * epsilon_vals**-2

    integral = np.sum(integrand_vals * delta_eps)
    return integral

def expnu_new(r, Enu, dec, search_params):
    """
    Expected count of neutrinos for given parameters.

    Parameters:
    -----------
    r: float
        Distance in Mpc units
    Enu: float
        Total energy of the emitted neutrinos by an astrophysical source, in erg units
    dec: float
        Declination angle in degrees
    search_params: IceCubeLIGO object
        Constant parameters used in this search
    
    Returns:
    --------
    Expected number of neutrinos to detect, in float
    """
    Mpc_to_cm = 3.085677581e24
    erg_to_GeV = 624.1509074

    Enu = Enu*erg_to_GeV
    r = r*Mpc_to_cm

    epsilon_bins = np.array(epsilon_dict()) 
    epsilon_vals = 10**((epsilon_bins[:-1] + epsilon_bins[1:]) / 2)
    delta_eps = 10**epsilon_bins[1:] - 10**epsilon_bins[:-1]
    Aeff_vals = [Aeff(epsilon, dec, search_params)*epsilon**-2 for epsilon in epsilon_vals ]

    energy_norm = 13.8 # ln(epsilonmax/epsilonmin)

    int_vals = Aeff_vals*delta_eps*Enu/(4.0*np.pi)/energy_norm*r**-2

    return np.sum(int_vals)


def PNnu(gw_skymap, Nnu, search_params):
    from scipy.stats import poisson

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
    total = np.sum(integral_vals)*delta_dec*delta_r
    return total


def expnu(r, Enu,  search_params):
    """Count of expected neutrinos."""
    return search_params.nu_51_100*(Enu/search_params.Enumax)*(100/r)**2


def Pempfar(far):
    """Probability of a false alarm rate in null hypothesis, obtained from the
    histogram of all superevents released in GraceDB from O1 to end of O4. Empirical
    probability of a false alarm rate.
    
    Parameters
    ----------
    far: float 
        False alarm rate in Hz
    """
    far_data = GW_BG_FARS
    if far > 2.3e-5:
        far = 2.3e-5
        
    log_bins_1 = np.logspace(-52, -10, num=(52-10)+1)

    log_bins_2 = np.logspace(-10, np.log10(2.3e-5), num=4 * (int(np.log10(2.3e-5)) + 10))

    bins = np.concatenate([log_bins_1, log_bins_2])
    bins = np.unique(bins)

    counts, bin_edges = np.histogram(far_data, bins=bins, density=True)
    bin_widths = np.diff(bin_edges)

    probabilities = counts * bin_widths

    if far < bin_edges[0] or far > bin_edges[-1]: # False alarm rate is out of range
        return 0
    
    bin_i = np.digitize(far, bin_edges) - 1

    return probabilities[bin_i] if 0 <= bin_i < len(probabilities) else 0


def t_overlap(tgw, tnu, search_params):
    return max(
        0,
        (min(tgw+search_params.tgwplus, tnu+search_params.tnuplus) -
         max(tgw+search_params.tgwminus, tnu+search_params.tnuminus))
    )


class IceCubeLIGO(BaseModel):
    """Collection of the search parameters."""
    nu_51_100: float = Field(..., description="Detector specific number for IceCube's neutrino detections.")
    tgwplus: float = Field(..., description="Upper limit for the GW detection time boundaries, in seconds.")
    tgwminus: float = Field(..., description="Lower limit for the GW detection time boundaries, in seconds.")
    tnuplus: float = Field(..., description="Upper limit for the neutrino detection time boundaries, in seconds.")
    tnuminus: float = Field(..., description="Lower limit for the neutrino detection time boundaries, in seconds.")
    fb: float = Field(..., description="Beaming factor for neutrinos.")
    ratebggw: float = Field(..., description="Background rates for the gravitational wave channels. Including every pipeline, in Hz.")
    ratebgnu: float = Field(..., description="Background neutrino detection rate in IceCube channels, obtained from IceCube data releases, in Hz.")
    ndotgw: float = Field(..., description="Observable gravitational wave rate, calculated in the integral in eq. (26)")
    ndotnu: float = Field(..., description="Observable astrophysical neutrino rate, calculated in the integral in eq. (29)")
    ndotgwnu: float = Field(..., description="Observable GW and neutrino multi-messenger rate.")
    Mgwmax: float = Field(..., description="""Maximum mass of one of the objects in a GW emitting event accounted by the model. Different
                          for the types of merger events released by LIGO's GraceDB. [kg]""")
    Mgwmin: float = Field(..., description="""Minimum mass of one of the objects in a GW emitting event accounted by the model. Different
                          for the types of merger events released by LIGO's GraceDB. [kg]""")
    Enumax: float = Field(..., description="Maximum isotropic-equivalent neutrino energy accounted by the model. [Ergs]")
    Enumin: float = Field(..., description="Minimum isotropic-equivalent neutrino energy accounted by the model. [Ergs]")
    epsilonmax: float = Field(..., description="Maximum neutrino energy for the integral upper bounds. [GeV]")
    epsilonmin: float = Field(..., description="Minimum detectible neutrino energy. [GeV]")
    farthr: float = Field(..., description="""False alarm rate threshold for gravitational wave detections. Events with false alarm
                          rate below this number are not detected by any pipelines. [Hz]""")
    population: str = Field(..., description="Name of the source population of gravitational wave events. Can be 'bns', 'bbh', 'nsbh'.")
    

def search_parameters(population):
    """Return an IceCubeLIGO instance with values ascribed into.
    population: "bbh", "bns", or "nsbh"
    """
    return IceCubeLIGO(
        nu_51_100 = 2.12,
        tgwplus = 250., # s
        tgwminus = -250., # s
        tnuplus = 250., # s
        tnuminus = -250., # s
        fb = 10.0, 
        ratebggw = 5.0*2.3*10**-5, # 1/s, 2 per day per pipeline (gstlal, mbta)
        ratebgnu = 0.0035055081034729364, # Background neurino rate, per second (1/s)
        ndotgw = 3.9255868248144146*3.16*10**-8, # per second
        ndotnu = 2.7402641832764836e-05*3.16*10**-8, # per second
        ndotgwnu = 2.7402641832764836e-05*3.16*10**-8/2.0, #FIXME currently ndotnu/2
        Mgwmax = 2.5*1.988*10**30, # 2.5 Solar Masses in Kgs for bns
        Mgwmin = 1.0*1.988*10**30, # 1.0 Solar Masses in Kgs for bns
        Enumax = 10**51, # erg
        Enumin = 10**46, # erg
        epsilonmax = 10.0**8, # GeV
        epsilonmin = 10.0**2, # GeV
        farthr = 2.3*10**-5, # Hz
        population=population
    )
    
def match_far(gpstime, distance, right_ascension, declination, mass1, mass2):
    """Finds a false alarm rate from the O3 Sensitivity measurements data set for the given parameters"""
    import astropy.units as u
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz
    from astropy.time import Time

    location = EarthLocation(lon=0 * u.deg, lat=0 * u.deg, height=0 * u.m)

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
            (gw_data["far_mbta"][index] <= 2) * gw_data["far_mbta"][index]
        ]
    )

class IceCubeNeutrino():
    """Class for IceCube's neutrino detections."""
    def __init__(self, mjd, ra, dec, sigma, epsilon):
        self.mjd = mjd
        self.ra = ra
        self.dec = dec
        self.sigma = sigma
        self.epsilon = epsilon

    # mjd: float = Field(..., description="Detection time of the neutrino in MJD units.")
    # ra: float = Field(..., description="Right ascension angle of the neutrino. [Deg]")
    # dec: float = Field(..., description="Declination angle of the neutrino. [Deg]")
    # epsilon: float = Field(..., description="Reconstructed energy of the neutrino. [GeV]")
    # sigma: float = Field(..., description="Uncertainty radius of neutrino detection. [Deg]")

    @property
    def gps(self):
        """Converts mjd time format into gps time used in GW detections."""
        import astropy.time
        return astropy.time.Time(self.mjd, format='mjd').gps
    
    def psf(self):
        """Returns PartialUniqSkymap of the neutrino detection as
        a Gaussian point spread function centered around the right
        ascension and declination with sigma uncertainty.
        
        Parameters
        ----------
        right_ascension: Right ascension angle of neutrino in floating
        number and degrees for unit.
        declination: Declination angle of neutrino in floating number and
        degrees for unit.
        sigma: Standard deviation provided by IceCube neutrino detection.
        """

        from hpmoc.psf import psf_gaussian
        return psf_gaussian(ra=self.ra, dec=self.dec, sigma=self.sigma)
    
    def full_skymap(self, skymap):
        """Full skymap as an array with same nside as the skymap parameter."""
        partialskymap = self.psf()
        return partialskymap.fill(nside=skymap.nside)
    
