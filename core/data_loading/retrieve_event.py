from ligo.gracedb.rest import GraceDb
from astropy.table import QTable
from astropy.utils.data import download_file

client = GraceDb()

def retrieve_event(event_name):
    """Retreaving the skymap from GraceDB and reading in Qtable format"""
    gw = client.superevent(event_name)
    gw_dict = gw.json()
    t_GW = gw_dict.get("t_0")
    far = gw_dict.get("far")
    files = client.files(event_name).json()
    if "bayestar.multiorder.fits" in files:
        skymap_url = files["bayestar.multiorder.fits"]
    elif "Bilby.multiorder.fits" in files:
        skymap_url = files["Bilby.multiorder.fits"]
    else:
        multiorder_maps = [s for s in files if s.endswith("multiorder.fits")]
        skymap_url = files[multiorder_maps[0]]
    filename = download_file(skymap_url, cache=True)

    skymap = QTable.read(filename)
    return skymap, t_GW, far
