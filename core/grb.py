"""GBM/GRB skymaps"""
# grb_real_url = "https://gcn.gsfc.nasa.gov/notices_gbm_sub/gbm_subthresh_780770594.088000_healpix.fits"
grb_real_url = "http://gcn.gsfc.nasa.gov/notices_f/gbm_gnd_loc_map_780958803.fits"
import numpy as np
from astropy.utils.data import download_file
from astropy.table import QTable
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import astropy.units as u
import healpy as hp

from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from reproject import reproject_to_healpix
from skymap import HealPixSkymap

filename = download_file(grb_real_url, cache=True)
# grb_skymap = QTable.read(filename)
hdul = fits.open(filename)

hpxmap, footprint = reproject_to_healpix(
    hdul['PMAP'],
    coord_system_out='icrs',
    nside=256,
    nested=True,
    hdu_in=1
)

print(hpxmap)
skymap = HealPixSkymap(hpxmap, moc=False)
print(skymap.pixels)
skymap.plot()

data = hdul[1].data
header = hdul[1].header
wcs = WCS(header)
print(wcs)
nside = 256
npix = hp.nside2npix(nside)
pixels = np.zeros(npix)


ny, nx = data.shape  # (512, 512)
y, x = np.mgrid[0:ny, 0:nx]  # pixel grid

# Convert to RA, Dec
pre_ra, pre_dec = wcs.all_pix2world(x, y, 0, ra_dec_order=True)  
ra, dec = pre_ra.ravel(), pre_dec.ravel()
vals = data.ravel()

print(ra,dec)

ipix = hp.ang2pix(256, ra, dec, nest=True, lonlat=True)
print(ipix)
print(f"length of pixels {len(ipix)}, length of pixel values {len(vals)}")

theta, phi = hp.pix2ang(256, np.arange(npix), nest=True)
nested_ra = np.degrees(phi)
nested_dec = 90.0 - np.degrees(theta)

print(f"length of ra values {len(nested_ra)}, length of dec values {len(nested_dec)}")

hpxmap = np.zeros(npix, dtype=np.float64)