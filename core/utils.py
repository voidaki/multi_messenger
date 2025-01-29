import pandas as pd

def save_dict_to_hdf5(data_dict, filepath):
    """
    Save a dictionary of Pandas DataFrames to an HDF5 file.
    
    Args:
        data_dict (dict): Dictionary where keys are dataset names and values are DataFrames.
        filepath (str): Path to the output HDF5 file.
    """
    with pd.HDFStore(filepath, mode="w") as store:
        for key, df in data_dict.items():
            store.put(key, df)  # Save each DataFrame under its key
    print(f"Saved {len(data_dict)} DataFrames to {filepath}")


def load_hdf5_to_dict(filepath):
    """
    Load an HDF5 file into a dictionary of Pandas DataFrames.
    
    Args:
        filepath (str): Path to the HDF5 file.
    
    Returns:
        dict: Dictionary where keys are dataset names and values are DataFrames.
    """
    data_dict = {}
    with pd.HDFStore(filepath, mode="r") as store:
        for key in store.keys():
            data_dict[key.strip("/")] = store[key]  # Remove leading '/'
    return data_dict

# Example Usage
loaded_data = load_hdf5_to_dict("processed_data.h5")
print(loaded_data["events"].head())  # Access the "events" DataFrame

