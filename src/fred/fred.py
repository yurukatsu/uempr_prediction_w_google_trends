import numpy as np
from full_fred.fred import Fred


def get_data(path_key:str, factor:str):
    fred = Fred(path_key)
    df = fred.get_series_df(factor)
    df = df.loc[:, ["date", "value"]]
    df["value"].replace(".", np.nan, inplace=True)
    df.dropna(how="any", inplace=True)
    df["value"] = df["value"].astype(float)
    df = df.rename(columns={"value":factor})
    return df