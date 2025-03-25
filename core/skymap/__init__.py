from hpmoc import PartialUniqSkymap
from data_loading import retrieve_event


def neutrinopsf(ra, dec, sigma):
    """Returns PartialUniqSkymap of the neutrino detection as
    a Gaussian point spread function centered around the right
    ascension and declination with sigma uncertainty.
    
    Parameters
    ----------
    right_ascension: Right ascension angle of neutrino in degrees for unit.
    declination: Declination angle of neutrino in floating number and
    degrees for unit.
    sigma: Standard deviation provided by IceCube neutrino detection."""

    from hpmoc.psf import psf_gaussian

    return psf_gaussian(ra, dec, sigma)


def emptyskymap(p, skymap):
    """Returns a skymap with with uniform value p, shaped in the nest
    of entered skymap.
    
    Parameters
    ----------
    p: int or float
        The value each pixel of the skymap is set at.
    skymap: Skymap
        Used to create the nesting of the new skymap.
    """

    

