import numpy as np
from scipy.stats import norm, uniform
import astropy.units as u
from astropy.utils.data import download_file
import astropy_healpix as ah
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import pandas as pd
from tqdm import tqdm

from data_loading import load_gravitational_wave_data

gw_data = load_gravitational_wave_data()
print(gw_data)
