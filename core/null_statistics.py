"""
author:voidaki
"""
from pathlib import Path
import numpy as np
import random
from astropy.time import Time
import matplotlib.pyplot as plt
from tqdm import tqdm
import bisect
import sys

from coincidence_sig import *

EMP_NU = np.load("/home/aki/snakepit/multi_messenger_astro/data/neutrino_data/emprical_neutrinos.npy") # each element is np.array(epsilon (log10(e/GeV)), ra, dec, sigma)

LVK_skymap_folders = [Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4a"),
                      Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4b"),
                      Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4c")]

LVK_skymap_paths = [file for folder in LVK_skymap_folders
                    for file in folder.iterdir()]

false_alarm_rate_path = Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4_fars")

start_gps = Time('2024-01-01T00:00:00', scale='utc').gps
end_gps   = Time('2025-01-01T00:00:00', scale='utc').gps

start_mjd = Time('2024-01-01T00:00:00', scale='utc').mjd
end_mjd = Time('2025-01-01T00:00:00', scale='utc').mjd

burst_pipelines = ['cwb', 'mly', 'olib']

search_params = search_parameters("bns")

def random_gw_event():
    gw_skymap_path = random.choice(LVK_skymap_paths)
    graceid = gw_skymap_path.name.split('_')[0]
    pipeline = gw_skymap_path.name.split('_')[1].replace('.multiorder.fits', '')

    if pipeline in burst_pipelines:
        burst = True
    else:
        burst = False
    
    far_path = false_alarm_rate_path / (graceid + "_far.npy")
    far = np.load(far_path)
    tgw = np.random.uniform(start_gps, end_gps)
    gw_skymap = HealPixSkymap.load_locally(gw_skymap_path, burst=burst, title=f"{graceid} Gravitational Wave Skymap with Neutrino Coincidences")
    return tgw, float(far), gw_skymap, burst

def random_nu_event():
    nu = random.choice(EMP_NU)
    nu_mjd = np.random.uniform(start_mjd, end_mjd)
    return IceCubeNeutrino(nu_mjd, nu[1], nu[2], nu[3], 10.0**nu[0])

def generate_null_statistics(Nevents,index):
    random_generated_neutrinos = []
    null_stat = []
    for i in tqdm(range(Nevents), desc=f'Generating random events for run {index}'):
        random_generated_neutrinos.append(random_nu_event())
    
    random_generated_neutrinos.sort(key=lambda n: n.gps)
    gps_list = [n.gps for n in random_generated_neutrinos]
    count = 0

    while count < 80:
        tgw, far, gw_skymap, burst = random_gw_event()
        if burst:
            print("Gravitational wave was burst event so it was skipped!")
            print(">----------------------------------------------------<\n")
            continue
        skymap = gw_skymap.rasterize(as_skymap=True)
        
        left = bisect.bisect_left(gps_list, tgw + search_params.tnuminus)
        right = bisect.bisect_right(gps_list, tgw + search_params.tnuplus)

        neutrino_list = random_generated_neutrinos[left:right]
        print(f"tgw: {tgw}, far: {far:.3g}, skymap nside: {skymap.nside}, Nnu: {len(neutrino_list)}")
        if len(neutrino_list) == 0:
            test_statistic = 0.
        else:
            test_statistic = TS(tgw, skymap, far, neutrino_list)
        print(f"test staticstic with {len(neutrino_list)} neutrinos: {test_statistic:.3g}")
        print(">--------------------------------------------------------------------------<\n")
    
        # if test_statistic > 0.5e-21:
        #      skymap.plot(neutrino_list)
        #      plt.show()
        
        count += 1
        null_stat.append(test_statistic)

    return np.array(null_stat)

import os

for i in range(0, 500):
    filename = f"./noncwb/null_stat{i}.npy"
    
    if os.path.exists(filename):
        print(f"{filename} exists, skipping.")
        continue

    null_stat = generate_null_statistics(110450,i)
    np.save(filename, null_stat)
    print(f"{i}th null_statics was generated and saved!")
