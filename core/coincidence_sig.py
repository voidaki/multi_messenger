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
    IceCubeLIGO,
    search_parameters,
    t_overlap,
    IceCubeNeutrino
)
from skymap import (
    HealPixSkymap,
    emptyskymap,
    Aeff_skymap
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


def signal_likelihood(tgw, gw_skymap, far, neutrino_list, search_params=search_parameters("bns")): # FIXME add cbc check, if the superevent is burst or not
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
    cbc: bool
        Whether this is a CBC (compact binary coalescence) group trigger or a
        burst trigger. True for cbc group, false for unmodeled searches.

    Returns
    -------
    Signal hypothesis likelihood for candidate multi-messenger (GW&HEN) detections
        P(x|θ,H_s)P(θ|H_s) Integrated over all-sky and parameters
    """

    from scipy.integrate import nquad
    import astropy.units as u
    from scipy.stats import poisson

    if len(neutrino_list) == 0:
        return 0.
    
    pix_area = gw_skymap.nside2pixarea() # in u.sr
    nuskymap = emptyskymap(0.0, gw_skymap)
    Nnu = len(neutrino_list)
    Tobs = search_params.tgwplus - search_params.tgwminus

    def Pθ_Hs(Enu, r):
        "P(θ|H_s) for a single gravitational wave detection and Nnu number of neutrinos."
        PHs_θ = (poisson.pmf(0, search_params.ratebggw*Tobs)*poisson.pmf(0, search_params.ratebgnu*Tobs)*poisson.pmf(Nnu, expnu(r, Enu, search_params))*
                poisson.pmf(1, search_params.ndotgwnu*Tobs)*poisson.pmf(0, (search_params.ndotgw - search_params.ndotgwnu)*Tobs)*
                poisson.pmf(0, (search_params.ndotnu - search_params.ndotgwnu)*Tobs))
        return PEnu(Enu, search_params)*Pr(r, search_params)*sky_dist()*PHs_θ

    for neutrino in neutrino_list:
        a = emptyskymap(t_overlap(tgw, neutrino.gps(), search_params)*sky_dist()*nquad(Pθ_Hs, [(search_params.Enumin, search_params.Enumax), 
                                                                                               (0, 700.0)])[0], gw_skymap)
        nu = gw_skymap.neutrinoskymap(neutrino.ra, neutrino.dec, neutrino.sigma)
        nu = HealPixSkymap(nu.s, uniq=nu.u).rasterize(pad=0., as_skymap=True)
        nuskymap.pixels += (nu.pixels*pix_area).to(u.dimensionless_unscaled).value*a.pixels*Aeff_skymap(neutrino.epsilon, gw_skymap).pixels

    prob_dens = gw_skymap.pixels*nuskymap.pixels
    prob_map = (prob_dens*pix_area).to(u.dimensionless_unscaled).value*Pfar(far)
    
    allsky_integral = prob_map.sum()
    denominator = (search_params.tgwplus - search_params.tgwminus) * (search_params.tnuplus - search_params.tnuminus)

    return allsky_integral/denominator
    
def SLwogw(tgw, gw_skymap, far, neutrino_list, search_params=search_parameters("bns")):
    """Returns the likelihood without gravitational wave, ie. noise GW."""
    from scipy.stats import poisson
    from scipy.integrate import nquad
    import astropy.units as u

    pix_area = gw_skymap.nside2pixarea() # in u.sr
    nuskymap = emptyskymap(0.0, gw_skymap)
    Nnu = len(neutrino_list)
    Tobs = search_params.tnuplus - search_params.tnuminus

    def Pθ_H0nu(Enu, r):
        PHgw0_θ = (poisson.pmf(0, search_params.ratebgnu*Tobs)*poisson.pmf(1, search_params.ratebggw*Tobs)*
                   poisson.pmf(Nnu, expnu(Enu, r, search_params))*poisson.pmf(0,search_params.ndotgwnu*Tobs)*
                   poisson.pmf(0, (search_params.ndotgw-search_params.ndotgwnu)*Tobs)*
                   poisson.pmf(Nnu, (search_params.ndotnu - search_params.ndotgwnu)*Tobs))
        return PEnu(Enu, search_params)*Pr(r, search_params)*sky_dist()*PHgw0_θ

    for neutrino in neutrino_list:
        a = emptyskymap(sky_dist()*nquad(Pθ_H0nu, [(search_params.Enumin, search_params.Enumax), 
                                                    (0, 700.0)])[0], gw_skymap)
        nu = gw_skymap.neutrinoskymap(neutrino.ra, neutrino.dec, neutrino.sigma)
        nu = HealPixSkymap(nu.s, uniq=nu.u).rasterize(pad=0., as_skymap=True)
        nuskymap.pixels += (nu.pixels*pix_area).to(u.dimensionless_unscaled).value*a.pixels*Aeff_skymap(neutrino.epsilon, gw_skymap).pixels    

    allsky_integral = nuskymap.pixels.sum()*Pempfar(far)
    denominator = search_params.tgwplus - search_params.tgwminus

    return allsky_integral/denominator

def SLwonu(tgw, gw_skymap, far, neutrino_list, search_params=search_parameters("bns")):
    """Returns the likelihood without neutrino, ie. noise neutrino."""
    from scipy.stats import poisson
    from scipy.integrate import nquad

    Nnu = len(neutrino_list)
    Tobs = search_params.tgwplus - search_params.tgwminus

    def Pθ_Hgw0(Enu, r):
        PHgw0_θ = (poisson.pmf(Nnu, search_params.ratebgnu*Tobs)*poisson.pmf(0, search_params.ratebggw*Tobs)*
                   poisson.pmf(0, expnu(Enu,r, search_params))*poisson.pmf(0,search_params.ndotgwnu*Tobs)*
                   poisson.pmf(1, (search_params.ndotgw-search_params.ndotgwnu)*Tobs)*
                   poisson.pmf(0, (search_params.ndotnu - search_params.ndotgwnu)*Tobs))
        return PEnu(Enu, search_params)*Pr(r, search_params)*sky_dist()*PHgw0_θ

    null_nu_prob = 0
    for neutrino in neutrino_list:
        null_nu_prob += Pempe(neutrino.epsilon, neutrino.dec)*nquad(Pθ_Hgw0, [(search_params.Enumin, search_params.Enumax), 
                                                                                    (0, 700.0)])[0]
    allsky_integral = gw_skymap.allsky_integral()*Pfar(far)
    denominator = (search_params.tnuplus - search_params.tnuminus)
    
    return allsky_integral*null_nu_prob/denominator*Nnu**-1

def null_likelihood(far, neutrino_list, search_params=search_parameters("bns")):
    """Returns the null likelihood, ie. both detections are noise."""
    denominator = (search_params.tgwplus - search_params.tgwminus)*(search_params.tnuplus - search_params.tnuminus)

    Nnu = len(neutrino_list)
    nominator = 0
    for neutrino in neutrino_list:
        nominator += Pempe(neutrino.epsilon, neutrino.dec, search_params)

    return Pempfar(far)*nominator*Nnu**-1/denominator
    

def TS(tgw, gw_skymap, far, neutrino_list, search_params=search_parameters("bns")):
    nominator = signal_likelihood(tgw, gw_skymap, far, neutrino_list, search_params)*Phgwnu()
    denominator = (SLwogw(tgw, gw_skymap, far, neutrino_list, search_params)*Ph0nu() +
    SLwonu(tgw, gw_skymap, far, neutrino_list, search_params)*Phgw0() + 
    null_likelihood(far, neutrino_list, search_params)*Ph00())
    print(nominator)
    print(denominator)
    return nominator/denominator