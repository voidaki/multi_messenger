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


class GWSkyMap():
    def __init__(self, skymap, nest=True):
        """Take the skymap QTable and initialize it as a class.
        nest: boolean
            True for laterilization from UNIQ to NESTED"""
        import numpy as np
        import astropy_healpix as ah
        import healpy as hp

        self.uniq = np.array(skymap["UNIQ"])
        self.prob = skymap["PROBDENSITY"]
        self.distmu = skymap["DISTMU"]
        self.distsigma = skymap["DISTSIGMA"]
        self.distnorm = skymap["DISTNORM"]
        self.nest = nest

        level, ipix = ah.uniq_to_level_ipix(self.uniq)
        if nest:
            nside = ah.level_to_nside(level) # Lateralization from UNIQ to NESTED
            self.nside = nside
            self.ipix = ipix
        
        ra, dec = hp.pix2ang(self.nside, self.ipix,
                          nest=self.nest, lonlat=True)
        self.ra = np.array(ra)
        self.dec = np.array(dec)


    def nside2pixarea(self):
        import healpy as hp
        return hp.nside2pixarea(self.nside)


# def neutrinoskymap(ra, dec, sigma, skymap, normalize=True):
#     import healpy as hp
#     import numpy as np

#     nside = skymap.nside
#     npix = hp.nside2npix(nside)
#     print("npix:", npix, type(npix))
#     theta = np.deg2rad(90 - dec)
#     phi = np.deg2rad(ra)
#     vec = hp.ang2vec(theta, phi)

#     ipix = np.arange(npix)
#     vecs = hp.pix2vec(nside, ipix)
#     ang_sep = np.arccos(np.clip(np.dot(vec, vecs), -1, 1))
#     ang_sep_deg = np.rad2deg(ang_sep)

#     psf = np.exp(-0.5 * (ang_sep_deg / sigma)**2)
#     if normalize:
#         psf /= psf.sum()  # Normalize

#     psf_map_nested = hp.reorder(psf, inp='RING', out='NESTED')
#     return psf_map_nested

def neutrinoskymap(ra, dec, sigma, skymap, normalize=True):
    import healpy as hp
    import numpy as np

    ipix_nested = skymap.ipix
    nsides = skymap.nside
    # Get pixel vectors
    vecs = hp.pix2vec(nsides, ipix_nested, nest=True)
    vecs = np.stack(vecs, axis=-1)  # (N, 3)

    # Get neutrino direction vector
    theta = np.deg2rad(90 - dec)
    phi = np.deg2rad(ra)
    source_vec = hp.ang2vec(theta, phi)  # shape (3,)

    dots = np.clip(vecs @ source_vec, -1, 1)
    ang_sep_rad = np.arccos(dots)
    ang_sep_deg = np.rad2deg(ang_sep_rad)

    # Gaussian PSF
    psf = np.exp(-0.5 * (ang_sep_deg / sigma)**2)

    pixel_area_sr = hp.nside2pixarea(nsides)
    psf *= pixel_area_sr
    if normalize:
        psf /= psf.sum()  # Proper normalization

    return psf



def emptyskymap(val, skymap):
    import healpy as hp
    import numpy as np

    nside = skymap.nside
    npix = hp.nside2npix(nside)

    emptymap = np.full(npix, val, dtype=np.float64)
    emptymap = hp.reorder(emptymap, inp='RING', out='NESTED')
    return emptymap


def Aeffskymap(epsilon, skymap):
    import healpy as hp
    import numpy as np

    from utils import Aeff
    aeff = Aeff(epsilon)

