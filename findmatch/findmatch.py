# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:40:46 2024

@author: dvesk
"""

import numpy
import matplotlib.pyplot as plt
import h5py

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from astropy.time import Time

f=h5py.File('~/jupyter-notebook/multi_messenger_astro/endo3_mixture-LIGO-T2100113-v12.hdf5','r')#enter your location for this file (download from https://zenodo.org/records/7890437)
f=f['injections']

location = EarthLocation(    lon=0 * u.deg, lat=0 * u.deg, height=0 * u.m)
azlist=numpy.concatenate((numpy.load('azlist0.npy'),numpy.load('azlist1.npy'),numpy.load('azlist2.npy'),numpy.load('azlist3.npy')))
altlist=numpy.concatenate((numpy.load('altlist0.npy'),numpy.load('altlist1.npy'),numpy.load('altlist2.npy'),numpy.load('altlist3.npy')))

'''pa=numpy.linspace(0,1,100)
sel=[]
for p in pa:
    count=0
    index=numpy.argsort(numpy.abs(f['pastro_pycbc_hyperbank'][()]-p))[count]
    while(f['far_gstlal'][index]>365.24*2):
        count+=1
        index=numpy.argsort(numpy.abs(f['pastro_pycbc_hyperbank'][()]-p))[count]    
    sel.append(index)'''
'''selpre=numpy.random.choice(numpy.arange(len(azlist)),size=int(0.001*len(azlist)))
sel=[]
for index in selpre:
    #if(f['far_mbta'][index]<=365.24*2 or f['far_gstlal'][index]<=365.24*2 or f['far_pycbc_bbh'][index]<=365.24*2 or f['far_pycbc_hyperbank'][index]<=365.24*2):
    if(f['far_pycbc_hyperbank'][index]<=365.24*2):
        sel.append(index)'''

def match_p_astro(inc,ra,dec,mass1,mass2,dist,gpstime,chieff):
    #returns the maximum p_astro of the best matching injection that has <=2/day FAR
    #inputs: inclination of the binary, right ascension and declination of its position, source frame masses m1>=m2, luminosity distance, detection gps time, chieff spin parameter = (m1*s1z+m2*s2z)/(m1+m2)
    #input units: angular variables are in radians, masses in solar mass, distance in Mpc, time in seconds, chieff unitless
    if(mass1<mass2):
        print('mass1 cannot be less than mass2')
        return 'error'
    if(dist<=0):
        print('distance should be positive')
        return 'error'
    if(gpstime<=0):
        print('gpstime should be positive')
        return 'error'
    if(abs(inc)>numpy.pi):
        print('is inclination correct?')
        return 'error'
    if(abs(dec)>numpy.pi/2):
        print('is dec correct?')
        return 'error'
    if(abs(ra)>2*numpy.pi):
        print('is ra correct?')
        return 'error'
    t = Time(gpstime*u.s,format='gps')
    altaz = AltAz(obstime=t, location=location)
    target = SkyCoord(ra=ra*u.rad,dec=dec*u.rad, frame='icrs')
    target_alt_az = target.transform_to(altaz)
    alt=target_alt_az.alt.value*numpy.pi/180
    az=target_alt_az.az.value*numpy.pi/180
    print(az,alt)
    angdistpre=numpy.sin(altlist)*numpy.sin(alt)+numpy.cos(altlist)*numpy.cos(alt)*numpy.cos(azlist-az)
    angdistpre[angdistpre>1]=1
    angdistpre[angdistpre<-1]=-1
    angdist=numpy.arccos(angdistpre)

    fchirp= (f['mass1_source'][()]*f['mass2_source'][()])**0.6*(f['mass1_source'][()]+f['mass2_source'][()])**-0.2
    ts=((f['spin1z']*f['mass1_source'][()]+f['spin2z']*f['mass2_source'][()])/(f['mass1_source'][()]+f['mass2_source'][()])-chieff)**2/(0.2**2)+angdist**2/((5*numpy.pi/180)**2)+(numpy.cos(f['inclination'][()])-numpy.cos(inc))**2/0.1**2+(fchirp-(mass1*mass2)**0.6*(mass1+mass2)**-0.2)**2/(2*((mass1*mass2)**0.6*(mass1+mass2)**-0.2)/20)**2+(f['distance'][()]-dist)**2/(100*dist/1000)**2
    count=0
    index=numpy.argsort(ts)[count]
    '''while(index in sel):
        count+=1
        index=numpy.argsort(ts)[count]'''

    print(count)
    print(azlist[index],altlist[index],f['inclination'][index],f['mass1_source'][index],f['mass2_source'][index],f['distance'][index],f['spin1z'][index],f['spin2z'][index])
    print(angdist[index])
    print(f['pastro_gstlal'][index],f['pastro_mbta'][index],f['pastro_pycbc_bbh'][index],f['pastro_pycbc_hyperbank'][index])
    print(f['far_mbta'][index], f['far_gstlal'][index], f['far_pycbc_bbh'][index], f['far_pycbc_hyperbank'][index])
    while (f['far_mbta'][index]>365.24*2 and f['far_gstlal'][index]>365.24*2 and f['far_pycbc_bbh'][index]>365.24*2 and f['far_pycbc_hyperbank'][index]>365.24*2):
    #while(f['far_pycbc_hyperbank'][index]>365.24*2):
        count+=1
        if(count==len(azlist)):
            print('cannot find a good event')
            return 'error'
        else:
            index=numpy.argsort(ts)[count]
            '''while(index in sel):
                count+=1
                index=numpy.argsort(ts)[count]'''
        print(count)
        print(azlist[index],altlist[index],f['inclination'][index],f['mass1_source'][index],f['mass2_source'][index],f['distance'][index],f['spin1z'][index],f['spin2z'][index])
        print(angdist[index])
        print(f['pastro_gstlal'][index],f['pastro_mbta'][index],f['pastro_pycbc_bbh'][index],f['pastro_pycbc_hyperbank'][index])
        print(f['far_mbta'][index], f['far_gstlal'][index], f['far_pycbc_bbh'][index], f['far_pycbc_hyperbank'][index])
    #return(f['pastro_pycbc_hyperbank'][index])
    return numpy.max([(f['far_gstlal'][index]<=365.24*2)*f['pastro_gstlal'][index],(f['far_mbta'][index]<=365.24*2)*f['pastro_mbta'][index],(f['far_pycbc_bbh'][index]<=365.24*2)*f['pastro_pycbc_bbh'][index],(f['far_pycbc_hyperbank'][index]<=365.24*2)*f['pastro_pycbc_hyperbank'][index]])

'''res=[]
pastro_sel=[]
c=0
for index in sel:
    print(c)
    mass1=f['mass1_source'][index]
    mass2=f['mass2_source'][index]
    dec=f['declination'][index]
    ra=f['right_ascension'][index]
    dist=f['distance'][index]
    gps=f['gps_time'][index]
    inclination=f['inclination'][index]
    chieff=(f['spin1z'][index]*mass1+f['spin2z'][index]*mass2)/(mass1+mass2)
    pastro_sel.append(f['pastro_pycbc_hyperbank'][index])#,(f['far_mbta'][index]<=365.24*2)*f['pastro_mbta'][index],(f['far_pycbc_bbh'][index]<=365.24*2)*f['pastro_pycbc_bbh'][index],(f['far_pycbc_hyperbank'][index]<=365.24*2)*f['pastro_pycbc_hyperbank'][index]]))
    #pastro_sel.append(numpy.max([(f['far_gstlal'][index]<=365.24*2)*f['pastro_gstlal'][index],(f['far_mbta'][index]<=365.24*2)*f['pastro_mbta'][index],(f['far_pycbc_bbh'][index]<=365.24*2)*f['pastro_pycbc_bbh'][index],(f['far_pycbc_hyperbank'][index]<=365.24*2)*f['pastro_pycbc_hyperbank'][index]]))
    res.append(match_p_astro(inclination,ra,dec,mass1,mass2,dist,gps,chieff))
    c+=1
plt.scatter(pastro_sel,res)

for i in range(6,12):
    plt.hist((f['pastro_gstlal'][(f['far_gstlal'][()]<365.24*2)*(f['optimal_snr_net'][()]<i+1)*(f['optimal_snr_net'][()]>i)]),bins=numpy.linspace(0,1,20),histtype='step',label=str(i)+'<snr<'+str(i+1),density=True)
plt.legend(loc=9)
plt.xlabel('pastro gstlal')'''

inclination=0.5
ra=0
dec=0
mass1=30
mass2=20
dist=1000
gps=1393454069
spin1z=0.5
spin2z=0.2
chieff=(spin1z*mass1+spin2z*mass2)/(mass1+mass2)
match_p_astro(inclination,ra,dec,mass1,mass2,dist,gps,chieff)

