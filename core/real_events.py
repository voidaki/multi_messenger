import numpy as np
import glob, os
import json
from pathlib import Path
from astropy.time import Time

from coincidence_sig import TS, p_value, test_statistic
from skymap import *
from utils import IceCubeNeutrino

directory = '/home/aki/snakepit/multi_messenger_astro/core/noncwb'
lvk_skymaps = Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4b")
file_list = sorted(glob.glob(os.path.join(directory, '*.npy')))

# Load and concatenate
all_null_stats = np.concatenate([np.load(f) for f in file_list])
all_null_stats = all_null_stats[all_null_stats <= 1.0]

nu_mjd = Time(1414942185.917, format='gps').mjd
neutrino_list = [IceCubeNeutrino(nu_mjd, 210.20, 20.74, 1.84, 10**4.4)]
grace_id = 'S241106ba'
gw_skymap = HealPixSkymap.load_locally("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4b/S241106ba_bayestar.multiorder.fits").rasterize(as_skymap=True)
far = np.load(f"/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4_fars/{grace_id}_far.npy")
tgw = 1414942323.86

test_stat = test_statistic(tgw, gw_skymap, far, neutrino_list)
# pvalue = p_value(test_statistic, all_null_stats)
# print(f"Test statistic of the event: {test_statistic}\np-value: {pvalue}")
print(json.dumps(test_stat, indent=4))
# gw_skymap.plot(neutrino_list)

print("\n>-----------------------------------------------------------<\n")

tgw = 1419135971.25
gw_skymap = HealPixSkymap.load_locally(lvk_skymaps / "S241225c_Bilby.multiorder.fits").rasterize(as_skymap=True)
far = np.load(f"/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4_fars/S241225c_far.npy")
nu_mjd1, nu_mjd2 = Time(tgw-238.24, format='gps').mjd, Time(tgw-24.51, format='gps').mjd
neutrino_list = [IceCubeNeutrino(nu_mjd1, 235.45, -42.76, 0.43, 10**4.4), IceCubeNeutrino(nu_mjd2, 87.36, -1.75, 2.45, 10**4.4)]

test_stat = test_statistic(tgw, gw_skymap, far, neutrino_list)
# pvalue = p_value(test_statistic, all_null_stats)
# print(f"Test statistic of the event: {test_statistic}\np-value: {pvalue}")
print(json.dumps(test_stat, indent=4))
# gw_skymap.plot(neutrino_list)

print("\n>-----------------------------------------------------------<\n")

tgw = 1411770067.27
gw_skymap = HealPixSkymap.load_locally(lvk_skymaps / "S240930df_bayestar.multiorder.fits").rasterize(as_skymap=True)
far = np.load("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4_fars/S240930df_far.npy")
nu_mjd = Time(tgw-32.61, format='gps').mjd
neutrino_list = [IceCubeNeutrino(nu_mjd, 246.14, 64.44, 1.88, 10**4.4)]

test_stat = test_statistic(tgw, gw_skymap, far, neutrino_list)
# pvalue = p_value(test_statistic, all_null_stats)
# print(f"Test statistic of the event: {test_statistic}\np-value: {pvalue}")
print(json.dumps(test_stat, indent=4))
# gw_skymap.plot(neutrino_list)

print("\n>-----------------------------------------------------------<\n")

tgw = 1414552629.04
gw_skymap = HealPixSkymap.load_locally(lvk_skymaps / "S241102o_bayestar.multiorder.fits", title=f"S241102o Gravitational Wave Skymap with Neutrino Coincidences").rasterize(as_skymap=True)
far = np.load("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4_fars/S241102o_far.npy")
nu_mjd = Time(tgw-44.62, format='gps').mjd
neutrino_list = [IceCubeNeutrino(nu_mjd, 338.77, 42.96, 1.48, 10**4.4)]

test_stat = test_statistic(tgw, gw_skymap, far, neutrino_list)
# pvalue = p_value(test_statistic, all_null_stats)
# print(f"Test statistic of the event: {test_statistic}\np-value: {pvalue}")
print(json.dumps(test_stat, indent=4))

lvk_skymaps = Path("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4a")
gw_skymap = HealPixSkymap.load_locally(lvk_skymaps / "S231205c_bayestar.multiorder.fits", title="S231205c Gravitational Wave Skymap with Neutrino Coincidences").rasterize(as_skymap=True)
far = np.load("/home/aki/snakepit/multi_messenger_astro/data/gw_data/LVK_skymaps/o4_fars/S231205c_far.npy")
nu_mjd = Time(tgw-198.05, format='gps').mjd
neutrino_list = [IceCubeNeutrino(nu_mjd, 220.38, 26.40, 0.86, 10**4.4)]

test_stat = test_statistic(tgw, gw_skymap, far, neutrino_list)
# pvalue = p_value(test_statistic, all_null_stats)
# print(f"Test statistic of the event: {test_statistic}\np-value: {pvalue}")
print(json.dumps(test_stat, indent=4))
