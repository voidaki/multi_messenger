import pandas as pd
import re
from pathlib import Path


def neutrino_data():
    """Load and process the neutrino effective areas and event data
    from a .csv file and order them into two seperate dictionaries
    that have keys corresponding to the name of the same .csv file."""
    effectiveArea_path = Path("../data/neutrino_data/irfs")
    events_path = Path("../data/neutrino_data/events")
    dataframes_effectiveArea = {}
    dataframes_events = {}

# Separating the data files column names into their proper place
    for file_path in effectiveArea_path.glob("*effectiveArea.csv"):
        with open(file_path, "r") as file:
            first_row = file.readline().strip()

        column_names = re.split(r"\s{2,}", first_row)
        column_names = [name for name in column_names if name != "#"]
        df = pd.read_csv(file_path, sep=r"\s+", skiprows=1, header=None)
        df.columns = column_names
        key = file_path.stem
        dataframes_effectiveArea[key] = df

    for file_path in events_path.glob("*.csv"):
        with open(file_path, "r") as file:
            first_row = file.readline().strip()

        column_names = re.split(r"\s{2,}", first_row)
        column_names = [name for name in column_names if name != "#"]
        df = pd.read_csv(file_path, sep=r"\s+", skiprows=1, header=None)
        df.columns = column_names
        key = file_path.stem
        dataframes_events[key] = df
    
    return {"effective_areas": dataframes_effectiveArea, "events": dataframes_events}
