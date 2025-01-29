import numpy as np
import h5py

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from astropy.time import Time

injection_file = "~/snakepit/multi_messenger_astro/data/gw_data/O3_sensitivity/endo3_bnspop-LIGO-T2100113-v12.hdf5"

with h5py.File(injection_file, 'r') as f:
    N_draw = f.attrs["total_generated"]
    
    gpstime_source = f['injections/gps_time'][:] # s
    inc_source = f['injections/inclination'][:]*180/np.pi # radians to degrees (-90, 90)
    dec_source = f['injections/declination'][:]*180/np.pi # radians to degrees (-90, 90)
    ra_source = f['injections/right_ascension'][:]*180/np.pi # radians to degrees (0,360)

gpstime_source = np.array(gpstime_source)
inc_source = np.array(inc_source)
ra_source = np.array(ra_source)
dec_source = np.array(dec_source)

location = EarthLocation(lon=0*u.deg, lat=0*u.deg, height=0*u.m)
alt_array = np.zeros(len(ra_source)//4 + 3)
az_array = np.zeros(len(dec_source)//4 + 3)

count = 0
percentage = 0
N = len(gpstime_source)//4
# Run this for loop in the ranges range(N), range(N, 2*N), range(2*N, 3*N), range(3*N, len(gpstime_source))
# Name the files for each range, in order: alt_array0.npy, az_array0.npy; alt_array1.npy, az_array1.npy...
for index in range(3*N, len(gpstime_source)):
    count += 1
    gpstime = gpstime_source[index]
    inclination = inc_source[index]
    right_ascension = ra_source[index]
    declination = dec_source[index]

    t = Time(gpstime*u.s, format='gps')
    altaz = AltAz(obstime = t, location = location)
    observation = SkyCoord(ra = right_ascension*u.deg, dec = declination*u.deg, frame='icrs')
    observation_alt_az = observation.transform_to(altaz)
    alt = observation_alt_az.alt.value
    az = observation_alt_az.az.value

    alt_array[index - 3*N] = alt
    az_array[index - 3*N] = az
    if count*100//N == 1:
        count = 0
        percentage += 1
        print(f"The process is at {percentage}%")


np.save('alt_array3.npy', alt_array)
np.save('az_array3.npy', az_array)

print(alt_array, az_array)
