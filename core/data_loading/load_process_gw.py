import numpy as np
import h5py


def load_gravitational_wave_data():
    injection_file = "/home/aki/snakepit/multi_messenger_astro/data/gw_data/O3_sensitivity/endo3_bnspop-LIGO-T2100113-v12.hdf5"
    # injection_file = "endo3_bbhpop-LIGO-T2100113-v12.hdf5"
    
    gravitational_wave_data = {}
    keys = ["gpstime", "mass1", "mass2", "distance", "inclination", "right_ascension", "declination", "far_gstlal", "far_mbta", "far_pycbc_hyperbank", "far_pycbc_bbh", "altitude", "azimuth"]
    with h5py.File(injection_file, "r") as f:
        N_draw = f.attrs["total_generated"]
        gpstime_source = f["injections/gps_time"][:]  # s
        mass1_source = f["injections/mass1_source"][:]  # M_solar
        mass2_source = f["injections/mass2_source"][:]  # M_solar
        distance_source = f["injections/distance"][:]  # Mpc
        inc_source = (f["injections/inclination"][:] * 180 / np.pi)  # radians to degrees (-90, 90)
        dec_source = (f["injections/declination"][:] * 180 / np.pi)  # radians to degrees (-90, 90)
        ra_source = (f["injections/right_ascension"][:] * 180 / np.pi)  # radians to degrees (0,360)
        # far_cwb = f['injections/far_cwb'][:] # Off for the bns systems
        far_gstlal = f["injections/far_gstlal"][:] / 365.25  # per day
        far_mbta = f["injections/far_mbta"][:] / 365.25  # per day
        far_pycbc_hyperbank = f["injections/far_pycbc_hyperbank"][:] / 365.25  # per day
        far_pycbc_bbh = f["injections/far_pycbc_bbh"][:] / 365.25  # per day
  
    alt_source = np.concatenate(
        (
            np.load("/home/aki/snakepit/multi_messenger_astro/data/gw_data/alt_array0.npy"),
            np.load("/home/aki/snakepit/multi_messenger_astro/data/gw_data/alt_array1.npy"),
            np.load("/home/aki/snakepit/multi_messenger_astro/data/gw_data/alt_array2.npy"),
            np.load("/home/aki/snakepit/multi_messenger_astro/data/gw_data/alt_array3.npy"),
        )
    )
    az_source = np.concatenate(
        (
            np.load("/home/aki/snakepit/multi_messenger_astro/data/gw_data/az_array0.npy"),
            np.load("/home/aki/snakepit/multi_messenger_astro/data/gw_data/az_array1.npy"),
            np.load("/home/aki/snakepit/multi_messenger_astro/data/gw_data/az_array2.npy"),
            np.load("/home/aki/snakepit/multi_messenger_astro/data/gw_data/az_array3.npy"),
        )
    )

    data_np = [
        np.array(gpstime_source),
        np.array(mass1_source),
        np.array(mass2_source),
        np.array(distance_source),
        np.array(inc_source),
        np.array(ra_source),
        np.array(dec_source),
        np.array(far_gstlal),
        np.array(far_mbta),
        np.array(far_pycbc_hyperbank),
        np.array(far_pycbc_bbh),
        alt_source,
        az_source]

    for i in range(len(keys)):
        key = keys[i]
        gravitational_wave_data[key] = data_np[i]

    return gravitational_wave_data