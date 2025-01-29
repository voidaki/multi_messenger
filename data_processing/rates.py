import numpy as np
from pathlib import Path
import re
import pandas as pd

if __name__ == "__main__":
    effectiveArea_path = Path("../data/neutrino_data/irfs")
    events_path = Path("../data/neutrino_data/events")
    dataframes_effectiveArea = {}
    dataframes_events = {}

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

   # Neutrinos detection rate
    times1 = np.empty(1)
    for key in dataframes_events.keys():
        df = dataframes_events[key]
        days = df["MJD[days]"]
        days = np.array(days)
        times1 = np.concatenate((times1, days))
    times = np.delete(times1, 0)
    print((np.max(times) - np.min(times)) / 365.25)
    rate = len(times) / ((np.max(times) - np.min(times)) * 24 * 60 * 60)    # Neutrino detection rate per second
    print(rate)

    np.save("rate.npy", np.array([rate]))
