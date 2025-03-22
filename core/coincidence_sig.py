from utils import (
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


def signal_likelihood(tgw, gw_skymap, r, far, neutrino):
    return 1

def TS(search_params=search_parameters("bns")):
    nominator = signal_likelihood()*Phgwnu()
    denominator = Ph0nu() + Phgw0() + Ph00()
    return nominator/denominator