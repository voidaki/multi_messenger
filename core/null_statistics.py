from pathlib import Path
import numpy as np
from astropy.time import Time

from coincidence_sig import *

EMP_NU = np.load("/home/aki/snakepit/multi_messenger_astro/data/neutrino_data/emprical_neutrinos.npy") # each element is np.array(epsilon, ra, dec, sigma)

LVK_skymap_folders = [Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4a"),
                      Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4b"),
                      Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4c")]

LVK_skymap_paths = [file for folder in LVK_skymap_folders
                    for file in folder.iterdir()]

false_alarm_rate_path = Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4_fars")

start_gps = Time('2024-01-01T00:00:00', scale='utc').gps
end_gps   = Time('2025-01-01T00:00:00', scale='utc').gps

def random_gw_event():
    pass

def random_nu_event():
    pass

def generate_null_statistics(Nevents):
    pass
