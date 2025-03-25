from utils import (
    Pr,
    sky_dist,
    PMgw,
    PEnu,
    Pfar,
    Aeff,
    expnu,
    Pempfar,
    matchfar,
    IceCubeLIGO,
    search_parameters
)


def Ph00(search_params=search_parameters("bns")):
    """Prior probability of null hypothesis of both detections
      being noise originated."""
    return search_params.ratebggw*search_params.ratebgnu

def Phgw0(search_params=search_parameters("bns")):
    """Prior probability of null hypothesis of real astrophysical
    gravitational wave detection with noise neutrino detection."""
    return search_params.ndotgw*search_params.ratebgnu

def Ph0nu(search_params=search_parameters("bns")):
    """Prior probability of null hypothesis noise related gravitational
    wave detection with real astrophysical neutrino detection."""
    return search_params.ratebggw*search_params.ndotnu

def Phgwnu(search_params=search_parameters("bns")):
    """Prior probability of signal hypothesis where both detections are
    coming from the same astrophysical source, ie. a multi-messenger event."""
    return search_params.ndotgwnu


def signal_likelihood(tgw, gw_skymap, far, neutrino_list, search_params=search_parameters("bns")):
    """Returns the signal likelihood in eq (3) 
    
    Parameters
    ----------
    tgw: float 
        Detection gps time of the gravitational wave, in s
    gw_skymap: multiorder.fits
        Skymap provided by GraceDB superevent release
    far: float 
        False alarm rate provided by GraceDB superevent release, in Hz
    neutrino_list: list
        List of all neutrinos in the time frame, as IceCubeNeutrino instance
    search_params: Collection of constant search parameters for this model.

    Returns
    -------
    Signal hypothesis likelihood for candidate multi-messenger (GW&HEN) detections
        P(x|θ,H_s)P(θ|H_s)
    """

    from scipy.integrate import nquad
    from scipy.stats import poisson
    import numpy as np

    if len(neutrino_list) == 0:
        return 0.
    
    def integrant():
        return 0
    
def SLwogw():
    """Returns the signal likelihood without gravitational wave, ie. noise GW."""
    return 1

def SLwonu():
    """Returns the signal likelihood without neutrino, ie. noise neutrino."""
    return 1

def null_likelihood():
    """Returns the null likelihood, ie. both detections are noise."""
    return 1
    

def TS(search_params=search_parameters("bns")):
    nominator = signal_likelihood()*Phgwnu()
    denominator = Ph0nu() + Phgw0() + Ph00()
    return nominator/denominator