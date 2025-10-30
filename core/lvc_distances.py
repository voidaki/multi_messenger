from skymap import HealPixSkymap
from utils import IceCubeNeutrino
from hpmoc.plotters import mollview, PointsTuple
import hpmoc
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from astropy.time import Time
from utils import expnu_new, search_parameters

gw_path = Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4c/S250201a_bayestar.multiorder.fits")
gw_skymap = hpmoc.PartialUniqSkymap.read(gw_path, strategy='ligo')
distnorm = hpmoc.PartialUniqSkymap.read(gw_path, strategy='basic', name='DISTNORM')
distmu = hpmoc.PartialUniqSkymap.read(gw_path, strategy='basic', name='DISTMU')
distsigma = hpmoc.PartialUniqSkymap.read(gw_path, strategy='basic', name='DISTSIGMA')
tgw = 0.0
neutrino_list = [IceCubeNeutrino(Time(tgw-1.1, format='gps').mjd, 224.5, 59.1, 0.5, 4.2*10**3.2), 
                 IceCubeNeutrino(Time(tgw+1.3, format='gps').mjd, 120.2, 62.9, 0.8, 3.3*10**3.5)]

points = [(neutrino.ra, neutrino.dec, neutrino.sigma) for neutrino in neutrino_list]
neutrino_points = PointsTuple(points, label=(f"neutrino {i}" for i in range(len(neutrino_list))))
gw_skymap.plot(neutrino_points)

def distance_integral(dnorm=distnorm.s, dmu=distmu.s, dsigma=distsigma.s):
    mask_mu = (~np.isinf(dmu)) & (dmu >= 0)
    norm = dnorm[mask_mu]
    mu = dmu[mask_mu]
    sigma = dsigma[mask_mu]
    sigma_2 = 2*sigma**2

    dec_angles = (distnorm.coords()[1][mask_mu]).value
    print(len(dec_angles))
    def gaussian(r):
        return norm*np.exp((r - mu)**2/sigma_2)
    
    print(f"{norm}\n{mu}\n{sigma}")
    return np.sum(sigma*(expnu_new(mu, 10**47, dec_angles, search_parameters("bns"))*gaussian(mu) 
                         + expnu_new(mu+sigma, 10**47, dec_angles, search_parameters("bns"))*2*gaussian(mu+sigma) 
                         + expnu_new(mu+2*sigma, 10**47, dec_angles, search_parameters("bns"))*2*gaussian(mu+2*sigma) 
                         + expnu_new(mu+3*sigma, 10**47, dec_angles, search_parameters("bns"))*gaussian(mu+3*sigma)))

now = time.time()
integral = distance_integral()
end = time.time()
import sys
# np.set_printoptions(threshold=sys.maxsize)
print(f"Integration value is: {integral}, time it took: {end-now}")
