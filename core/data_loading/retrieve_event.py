"""
author:voidaki
"""
from ligo.gracedb.rest import GraceDb
from astropy.table import QTable
from astropy.utils.data import download_file

client = GraceDb()

def retrieve_event(event_name):
    """Retreaving the skymap, event time, and false alarm rate
    from GraceDB and reading in Qtable format"""
    gw = client.superevent(event_name)
    gw_dict = gw.json()
    t_GW = gw_dict.get("t_0")
    far = gw_dict.get("far")
    files = client.files(event_name).json()
    if "Bilby.multiorder.fits" in files:
        skymap_url = files["Bilby.multiorder.fits"]
        stype = "Bilby"
    elif "LALInference.multiorder.fits" in files:
        skymap_url = files["LALInference.multiorder.fits"]
        stype = "LALInference"
    elif "bayestar.multiorder.fits" in files:
        skymap_url = files["bayestar.multiorder.fits"]
        stype = "bayestar"
    elif "cwb.multiorder.fits" in files:
        skymap_url = files["cwb.multiorder.fits"]
        stype = "cwb"
    elif "mly.multiorder.fits" in files:
        skymap_url = files["mly.multiorder.fits"]
        stype = "mly"
    elif "olib.multiorder.fits" in files:
            skymap_url = files["olib.multiorder.fits"]
            stype = "olib"
    else:
        multiorder_maps = [s for s in files if s.endswith("multiorder.fits")]
        skymap_url = files[multiorder_maps[0]]
    filename = download_file(skymap_url, cache=True)

    save_name = f"{event_name}_{stype}.multiorder.fits"
    
    skymap = QTable.read(filename)
    
    # skymap.write(save_name, overwrite=True)
    return skymap, t_GW, far
