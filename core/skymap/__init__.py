
class HealPixSkymap():
    def __init__(self, pixels, distmu, distsigma, distnorm, uniq=None, moc=True, nest=True):
        """
        A HealPix Skymap object, written to typically handle the multi-order
        skymaps provided by LIGO-VIRGO-KARGA colab. Contains the functions
        needed to manipulate, merge, multiply and create skymaps.
        """
        import numpy as np
        import astropy_healpix as ah

        self.pixels = pixels # 1 / sr
        self.distmu = distmu
        self.distsigma = distsigma
        self.distnorm = distnorm
        self.moc = moc
        self.nest = nest

        if moc:
            self.uniq = uniq
            level, ipix = ah.uniq_to_level_ipix(self.uniq)
            nside = ah.level_to_nside(level)
            self.nside = nside
            self.ipix = ipix
        if not moc:
            import healpy as hp
            npix = len(self.pixels)
            self.nside = hp.npix2nside(npix)
            self.ipix = np.arange(npix)

    # def rasterize(self):
    #     from hpmoc import PartialUniqSkymap
    #     nuniq_skymap  = PartialUniqSkymap(self.prob, self.uniq)
    #     nested_skymap = nuniq_skymap.fill(as_skymap=True)
    #     self.nside = max(self.nside)
    #     return nested_skymap

    def readQtable(skymap, moc=True):
        import numpy as np

        pixels = skymap["PROBDENSITY"] # 1 / sr
        distmu = skymap["DISTMU"]
        distsigma = skymap["DISTSIGMA"]
        distnorm = skymap["DISTNORM"]
        if moc:
            uniq = np.array(skymap["UNIQ"])

        return HealPixSkymap(pixels, distmu, distsigma, distnorm, uniq)

    def load_from_graceid(graceid):
        pass

    def nside2npix(self):
        return 12 * self.nside**2

    def nside2ang(self):
        import healpy as hp
        import numpy as np
        ra, dec = hp.pix2ang(self.nside, self.ipix,
                          nest=self.nest, lonlat=True)

        return np.array(ra), np.array(dec)
    
    def uniq2nested(self): # FIXME
        pass

    def rasterize(self, nest=True, as_skymap=False):
        """
        Rasterize a NUNIQ Healpix multi-order skymap into all-sky skymap
        with a constant nside value obtained from the highest resolution
        pixel from the nside array of the multi-order skymap. If there are
        pixels without a value, they are replaced with hp.UNSEEN

        Returns
        -------
        nested_skymap: array-like, if as_skymap is set False
            HealPixSkymap instance if as_skymap is set True
        """
        from hpmoc.utils import fill
        import astropy.units as u
        max_nside = max(self.nside)
        nested_skymap = fill(self.uniq, self.pixels, max_nside)
        if self.moc:
            self.distmu = fill(self.uniq, self.distmu, max_nside)
            self.distsigma = fill(self.uniq, self.distsigma, max_nside)
            self.distnorm = fill(self.uniq, self.distnorm, max_nside)
        if not as_skymap:
            return nested_skymap # as astropy.units.Quantity
        if as_skymap:
            return HealPixSkymap(nested_skymap, self.distmu, self.distsigma, self.distnorm, moc=False)
        
    def nside2area_per_pix(self):
        """Returns area per pixel, in 1/steradian if steradian is True
        in 1/deg^2 if its set False."""
        import healpy as hp
        import astropy.units as u
        
        area_per_deg2 = hp.nside2pixarea(self.nside, degrees=True)
        return area_per_deg2*u.deg**-2
    
    def allsky_integral(self):
        return sum(self.pixels)

    def steradian2deg(self, pixels = None):
        """
        Converts 1 / sr probability density skymap pixel array
        into 1 / deg^2 probability density.
        """
        import astropy.units as u
        return 1
    
    def partial_product(skymap1, skymap2):
        i = 0 # FIXME
        return skymap1[i]*skymap2[i]
    
    def neutrinoskymap(self, ra, dec, sigma, normalize=True):
        """Returns PartialUniqSkymap of the neutrino detection as
        a Gaussian point spread function centered around the right
        ascension and declination with sigma uncertainty. 
        Neurino's skymap is in the same shape and nside as the 
        skymap used to generate it.
        
        Parameters
        ----------
        ra: Right ascension angle of neutrino in degrees for unit.
        dec: Declination angle of neutrino in floating number and
            degrees for unit.
        sigma: Standard deviation provided by IceCube neutrino detection."""
        from hpmoc.psf import psf_gaussian
        if self.moc:
            return psf_gaussian(ra, dec, sigma, nside=max(self.nside)) # Create a PartialUniqSkymap, Point spread function map 
        if not self.moc:
            return psf_gaussian(ra, dec, sigma, nside=self.nside)

    def to_table(self):
        from astropy.table import QTable
        if self.moc:
            data = {'UNIQ': self.uniq, 'PIXELS': self.pixels, 'DISTMU': self.distmu, 'DISTSIGMA': self.distsigma, 'DISTNORM': self.distnorm}
        else:
            data = {'IPIX': self.ipix, 'PIXELS': self.pixels, 'DISTMU': self.distmu, 'DISTSIGMA': self.distsigma, 'DISTNORM': self.distnorm}
        return QTable(data)

    def plot(self):
        from hpmoc import PartialUniqSkymap

# class HealPixSkyMap():
#     def __init__(self, skymap, moc=True, nest=True):
#         """Take the skymap QTable and initialize it as a class.
#         moc: boolean
#             True for rasterization from UNIQ to NESTED"""
#         import numpy as np
#         import astropy_healpix as ah

#         self.uniq = np.array(skymap["UNIQ"])
#         self.prob = skymap["PROBDENSITY"]
#         self.distmu = skymap["DISTMU"]
#         self.distsigma = skymap["DISTSIGMA"]
#         self.distnorm = skymap["DISTNORM"]
#         self.nest = nest

#         level, ipix = ah.uniq_to_level_ipix(self.uniq)
#         if moc:
#             nside = ah.level_to_nside(level) # Lateralization from UNIQ to NESTED
#             self.nside = nside
#             self.ipix = ipix

#     def nside2ang(self):
#         import healpy as hp
#         import numpy as np
#         ra, dec = hp.pix2ang(self.nside, self.ipix,
#                           nest=self.nest, lonlat=True)

#         return np.array(ra), np.array(dec)

    
#     def neutrinoskymap(self, ra, dec, sigma, normalize=True):
#         from hpmoc.psf import psf_gaussian

#         nu_partial = psf_gaussian(ra, dec, sigma, nside=max(self.nside)) # Create a PartialUniqSkymap
#         nu_nested = nu_partial.fill(nside=max(self.nside),pad=1e-30, as_skymap=True) # Rasterize, UNIQ to NESTED
#         return nu_nested


def emptyskymap(val, skymap):
    import healpy as hp
    import numpy as np

    nside = skymap.nside
    npix = hp.nside2npix(nside)

    emptymap = np.full(npix, val, dtype=np.float64)
    emptymap = hp.reorder(emptymap, inp='RING', out='NESTED')
    return emptymap


def Aeff_skymap(epsilon, skymap):
    import healpy as hp
    import numpy as np

    from utils import Aeff
    aeff = Aeff(epsilon)

