from utils import (
    temporal,
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
    search_parameters,
    t_overlap
)
from skymap import (
    GWSkyMap,
    neutrinoskymap,
    emptyskymap,
    Aeffskymap
)
from likelihood import (
    Paeffe,
    Pempe
)

def Pthetagw(r, M1, M2):
    return temporal(search_parameters("bns"))*Pr(r, search_parameters("bns"))*sky_dist()*PMgw(M1, M2, search_params=search_parameters("bns"))

def Pthetanu(r, Enu):
    return temporal(search_parameters("bns"))*Pr(r, search_parameters("bns"))*sky_dist()*PEnu(Enu, search_params=search_parameters("bns"))

def Ptheta(r, Enu, M1, M2):
    return temporal(search_parameters("bns"))*Pr(r, search_parameters("bns"))*sky_dist()*PEnu(Enu, search_params=search_parameters("bns"))*PMgw(M1, M2, search_params=search_parameters("bns"))

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
    gw_skymap: HealPixSkymap instance
        Skymap provided by GraceDB superevent release
    far: float 
        False alarm rate provided by GraceDB superevent release, in Hz
    neutrino_list: list
        List of all neutrinos in the time frame, as IceCubeNeutrino instance
    search_params: Collection of constant search parameters for this model.

    Returns
    -------
    Signal hypothesis likelihood for candidate multi-messenger (GW&HEN) detections
        P(x|θ,H_s)P(θ|H_s) Integrated over allsky and time, parameters
    """

    from scipy.integrate import nquad
    from scipy.stats import poisson
    import numpy as np

    if len(neutrino_list) == 0:
        return 0.
    
    summednuskymap = emptyskymap(0, gw_skymap)
    count = -1
    for neutrino in neutrino_list:
        count += 1
        a = emptyskymap(t_overlap(tgw, neutrino.gps, search_params)*Paeffe(neutrino.epsilon, neutrino.dec)*sky_dist(), gw_skymap)
        summednuskymap += a*neutrinoskymap(neutrino.ra, neutrino.dec, neutrino.sigma, gw_skymap, normalize=False)
    
    def integrand(Enu, r):
        return 1
    
def SLwogw():
    """Returns the signal likelihood without gravitational wave, ie. noise GW."""
    return 1

def SLwonu():
    """Returns the signal likelihood without neutrino, ie. noise neutrino."""
    return 1

def null_likelihood(far, epsilon, dec, search_params=search_parameters("bns")):
    """Returns the null likelihood, ie. both detections are noise."""
    return temporal(search_params)**2*Pempfar(far)*Pempe(epsilon, dec)
    

def TS(search_params=search_parameters("bns")):
    nominator = signal_likelihood()*Phgwnu()
    denominator = SLwogw()*Ph0nu() + SLwonu()*Phgw0() + null_likelihood()*Ph00()
    return nominator/denominator