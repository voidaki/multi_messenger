import os
import tempfile
import json
import datetime
from typing import Any, Dict, List, Union
from pathlib import Path
import xmltodict
import xml.etree.ElementTree as ET
from pprint import pprint

TEXT_FILE_EXTENSIONS = (
    'txt',
    'xml',
    'csv',
    'json'
)

def save_voevent(voevent_bytes: bytes, filename: str):
    "VOEvent bytes can be obtained from the message.value, filename should be example.xml"
    with open(filename, "wb") as f:
            f.write(voevent_bytes)
            f.close()

def load_voevent(path: Path):
    tree = ET.parse(path)
    return tree.getroot()

def get_param(root: ET.Element, name):
    for param in root.findall(".//Param"):
        if param.attrib.get("name") == name:
            return param.attrib.get("value")
    return None
    
def get_iso_time(root: ET.Element):
    time_elem = root.find(".//ISOTime")
    if time_elem is not None:
        return time_elem.text
    return None

def get_RaDec(root: ET.Element):
    for position in root.find(".//Position2D"):
        pass
        

def parse_voevent_alert_to_xml_root(message_value):
    return ET.fromstring(message_value)

root = load_voevent(Path("/home/aki/snakepit/multi_messenger_astro/core/gcn_notices/2025-09-24 22:01:00.553096.xml"))
print(get_param(root, "GraceID"))
print(root.tag)
print(root.find(".//ISOTime").text)

print(root.find(".//C1").text)
print(root.find(".//C2").text)
print(root.find("How")[0].text)

class VOEvent():
    def __init__(self, directory: Union[str, Path] = "data"):
        self.directory = Path(directory)

    def parse_lvc(root: ET.Element):
        return {
            'source': 'LIGO Scientific Collaboration, Virgo Collaboration, and KAGRA Collaboration',
            'ivorn': root.attrib.get("ivorn"),
            'role': root.attrib.get("role"),
            'graceid': get_param(root, "GraceID"),
            'time': get_iso_time(root),
            'detectors': get_param(root, "Instruments"),
            'far': float(get_param(root, "FAR")),
            'significant': int(get_param(root, "Significant")),
            'burst': get_param(root, "Group") != "CBC",
            'pipeline': get_param(root, "Pipeline"),
            'skymap_url': get_param(root, "skymap_fits"),
            'bns': float(get_param(root, "BNS")),
            'nsbh': float(get_param(root, "NSBH")),
            'bbh': float(get_param(root, "BBH")),
            'terrestrial': float(get_param(root, "Terrestrial"))
        }

    def parse_fermi(root: ET.Element):
        return {
            'source': root.find(".//shortName").text,
            'ivorn': root.attrib.get("ivorn"),
            'role': root.attrib.get("role"),
            'time': get_iso_time(root),
            'detector': root.find("How")[0].text,
            'duration': float(get_param(root, "Trig_Timescale")),
            'snr': float(get_param(root, "Data_Signif")),
            'sun_ra': float(get_param(root, "Sun_RA")),
            'sun_dec': float(get_param(root, "Sun_Dec")),
            'ra': float(root.find(".//C1").text),
            'dec': float(root.find(".//C2").text),
            'sigma': float(root.find(".//Error2Radius").text)
        }
    

    def parse_icecube(icecube_path):
        pass 

lcv_voevent = load_voevent("/home/aki/snakepit/multi_messenger_astro/core/gcn_notices/2025-09-24 21:38:25.948210.xml")
pprint(VOEvent.parse_lvc(lcv_voevent))
pprint(VOEvent.parse_fermi(root))