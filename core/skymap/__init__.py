
class HealPixSkymap():
    def __init__(self, pixels, distmu=None, distsigma=None, distnorm=None, uniq=None, moc=True, nest=True, title=None):
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
        self.uniq = uniq
        self.title = title
        
        if moc:
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

        return HealPixSkymap(pixels, distmu, distsigma, distnorm, uniq, moc=moc)

    def load_from_graceid(graceid, moc=True):
        from data_loading import retrieve_event
        import numpy as np

        skymap = retrieve_event(graceid)[0]

        pixels = skymap["PROBDENSITY"] # 1 / sr
        distmu = skymap["DISTMU"]
        distsigma = skymap["DISTSIGMA"]
        distnorm = skymap["DISTNORM"]
        if moc:
            uniq = np.array(skymap["UNIQ"])

        return HealPixSkymap(pixels, distmu, distsigma, distnorm, uniq, moc=moc)

    def load_locally(filepath, burst=False, moc=True, title=None):
        """
        Loads skymap into HealPixSkymap instance from the filepath.
        Filepath should be a in multiorder.fits or multiorder.fits.gz
        format.

        Parameters:
        -----------
        filepath: str
            File path
        burst: bool
            True for cbc (non-burst) events, False for burst events.
        moc: bool
            True for Multi Order Coverage skymaps, False for Nester or RING ordering.

        Returns:
        --------
        HealpixSkymap instance
        """
        from astropy.table import QTable
        import numpy as np

        skymap = QTable.read(filepath)

        if not burst:
            pixels = skymap["PROBDENSITY"] # 1 / sr
            distmu = skymap["DISTMU"]
            distsigma = skymap["DISTSIGMA"]
            distnorm = skymap["DISTNORM"]
            if moc:
                uniq = np.array(skymap["UNIQ"])

            return HealPixSkymap(pixels, distmu, distsigma, distnorm, uniq, moc=moc, title=title)
        
        if burst:
            pixels = skymap["PROBDENSITY"]
            if moc:
                uniq = np.array(skymap["UNIQ"])
            return HealPixSkymap(pixels, uniq=uniq, moc=moc, title=title)

    def nside2npix(self):
        return 12 * self.nside**2

    def nside2ang(self):
        import healpy as hp
        import numpy as np
        ra, dec = hp.pix2ang(self.nside, self.ipix,
                          nest=True, lonlat=True)

        return np.array(ra), np.array(dec)
    
    def ipix2uniq(self): 
        import numpy as np
        return self.ipix + np.full(len(self.pixels), 4*self.nside**2, dtype=int)

    def rasterize(self, nside=None, pad=None, nest=True, as_skymap=False):
        """
        Rasterize a NUNIQ Healpix multi-order skymap into all-sky skymap
        with a constant nside value obtained from the highest resolution
        pixel from the nside array of the multi-order skymap. If there are
        pixels without a value, they are replaced with hp.UNSEEN or pad if
        pad is not None.

        Returns
        -------
        nested_skymap: array-like, if as_skymap is set False
            HealPixSkymap instance if as_skymap is set True
        """
        from hpmoc.utils import fill
        
        if nside is None:
            nside = max(256, max(self.nside))

        if pad is None:
            nested_skymap = fill(self.uniq, self.pixels, nside)
        else:
            nested_skymap = fill(self.uniq, self.pixels, nside, pad=pad)
        if self.distmu is not None:
            self.distmu = fill(self.uniq, self.distmu, nside)
            self.distsigma = fill(self.uniq, self.distsigma, nside)
            self.distnorm = fill(self.uniq, self.distnorm, nside)
        if not as_skymap:
            return nested_skymap # as astropy.units.Quantity
        if as_skymap:
            return HealPixSkymap(nested_skymap, self.distmu, self.distsigma, self.distnorm, uniq=self.ipix2uniq(),  moc=False, title=self.title)
        
    def nside2pixarea(self):
        """Returns area per pixel, in 1/steradian if steradian is True
        in 1/deg^2 if its set False."""
        import healpy as hp
        import astropy.units as u

        return hp.nside2pixarea(self.nside)*u.sr
    
    def allsky_integral(self):
        import healpy as hp
        import astropy.units as u
        pix_area = hp.nside2pixarea(self.nside)*u.sr
        prob_map = self.pixels*pix_area
        prob_map = prob_map.to(u.dimensionless_unscaled).value
        return prob_map.sum()
    
    
    def distance_gaussian_average(self, r):
        import numpy as np
        from scipy.stats import norm
        import astropy.units as u

        distribution = self.distnorm*norm.pdf(r, loc=self.distmu, scale=self.distsigma)
        distribution = distribution[distribution < 1.0*u.Mpc**-2]
        return (np.mean(distribution)*u.Mpc**2).to(u.dimensionless_unscaled).value


    def distance_pdf(self, ra, dec, r):
        """
        Returns the normal distribution of the distance from the gravitational wave
        skymap to 
        """
        from scipy.stats import norm
        import healpy as hp
        import astropy.units as u
        ipix = hp.ang2pix(self.nside, ra, dec, nest=True, lonlat=True)
        return (self.distnorm[ipix]*norm.pdf(r, loc=self.distmu[ipix], scale=self.distsigma[ipix])).to_value(u.Mpc**-2)

    # def skymap_integral(gwskymap, neutrino_list):
    #     import healpy as hp
    #     import astropy.units as u

    #     pix_area = hp.nside2pixarea(gwskymap.nside)*u.sr
    #     nuskymap = emptyskymap(0.0, gwskymap)
    #     for neutrino in neutrino_list:
    #         a = gwskymap.neutrinoskymap(neutrino.ra, neutrino.dec, neutrino.sigma)
    #         a = HealPixSkymap(a.s, uniq=a.u).rasterize(pad=0., as_skymap=True)
    #         nuskymap.pixels += (a.pixels*pix_area).to(u.dimensionless_unscaled).value
    #     prob_dens = gwskymap.pixels*nuskymap.pixels
    #     prob_map = (prob_dens*pix_area).to(u.dimensionless_unscaled).value
    #     return prob_map.sum()


    def steradian2deg(self, pixels = None): #FIXME
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
        if self.distmu is not None:
            if self.moc:
                data = {'UNIQ': self.uniq, 'PIXELS': self.pixels, 'DISTMU': self.distmu, 'DISTSIGMA': self.distsigma, 'DISTNORM': self.distnorm}
            else:
                data = {'IPIX': self.ipix, 'PIXELS': self.pixels, 'DISTMU': self.distmu, 'DISTSIGMA': self.distsigma, 'DISTNORM': self.distnorm}
        else:
            if self.moc:
                data = {'UNIQ': self.uniq, 'PIXELS': self.pixels}
            else:
                data = {'IPIX': self.ipix, 'PIXELS': self.pixels}
        return QTable(data)

    def plot(self, neutrino_list=None):
        import matplotlib.pyplot as plt
        if neutrino_list is None:
            from hpmoc import plot
            if self.moc:
                plot.plot((self.pixels, self.uniq))
            else:
                plot.plot(self.pixels)
        else:
            from hpmoc.plotters import mollview, PointsTuple
            import warnings
            warnings.filterwarnings("ignore",message=".*edgecolor.*for an unfilled marker.*")
            points = [(neutrino.ra, neutrino.dec, neutrino.sigma) for neutrino in neutrino_list]
            neutrino_points = PointsTuple(points, label=(f"neutrino {i}" for i in range(len(neutrino_list))))
            mollview(self.pixels, neutrino_points, rot=(180.0, 0., 0.), title=self.title)
        plt.show()

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
    emptymap = np.full((npix,), val, dtype=np.float64)
    emptymap = hp.reorder(emptymap, inp='RING', out='NESTED')

    return HealPixSkymap(emptymap, moc=False)


def Aeff_skymap(epsilon, skymap=None):
    import numpy as np
    from pathlib import Path
    import astropy.units as u
    from utils import epsilon_dict

    for i in range(41):
        if np.log10(epsilon) >= epsilon_dict()[40-i]:
            epsilon_index = 40-i
            break
        else:
            epsilon_index = 0
    aeff_filepath = Path("/home/aki/snakepit/multi_messenger_astro/data/neutrino_data/aeff_skymaps") / f"effective_area{epsilon_index}.npy"
    s = np.load(aeff_filepath)
    normalization= HealPixSkymap(s*epsilon**-2*u.sr**-1, moc=False).allsky_integral()
    aeff_skymap = HealPixSkymap(s*epsilon**-2/normalization, moc=False)
    
    if skymap is not None:
        if aeff_skymap.nside == skymap.nside:
            return aeff_skymap
        elif aeff_skymap.nside < skymap.nside:
            import healpy as hp
            pix = hp.ud_grade(aeff_skymap.pixels, nside_out=skymap.nside, order_in='NESTED', order_out='NESTED', power=0.0)
            return HealPixSkymap(pix, moc=False)
    else:
        return aeff_skymap

