import importlib.util
from chardet import detect
import subprocess
import os
from ucimlrepo import fetch_ucirepo
import pandas as pd

def check_encoding(file: str)-> str:
    with open(file, 'rb') as f:
        encoding = detect(f.read(10000))['encoding']
        return encoding

def load_apartments_data():
    """
    Checks if the file 'apartment_data.csv' exists in the current working directory.
    If not, fetches data using ucimlrepo, saves it as a CSV file, and loads it into a DataFrame.
    Returns a DataFrame `raw_df_apartments`.
    """
    file_path = r"./ML Predicting apartment prices v2/data/raw/raw_apartment_data.csv"

    if not os.path.exists(file_path):
        apartment_for_rent_classified = fetch_ucirepo(id=555)

        X = apartment_for_rent_classified.data.features
        y = apartment_for_rent_classified.data.targets

        # Combining features and targets into one DataFrame
        apartments_source = pd.concat([X, y], axis=1)

        # Saving to CSV
        apartments_source.to_csv(file_path, index=False)
        print(f"Data has been fetched and saved to the file '{file_path}'.")
    else:
        print(f"The file '{file_path}' already exists. Loading data...")

    # Loading data into a DataFrame
    apartments_source = pd.read_csv(file_path, encoding=check_encoding(file_path),low_memory=False,
                                    encoding_errors = 'ignore',
                                    delimiter=',')
    return apartments_source
