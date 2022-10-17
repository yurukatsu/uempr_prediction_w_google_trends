import argparse
import datetime
import pickle
import sys
from pathlib import Path
from time import sleep
from typing import List, Literal, Union

import pandas as pd
from dateutil.relativedelta import relativedelta

sys.path.append("../../src/ftrends")
import data


def load(
    kw_list:str,
    cat:str,
    geo:str,
    freq:Literal["M", "W", "D"]="M",
    verbose:bool=True,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    if freq == "M":
        df = data.load_monthly(kw_list, cat, geo)
        return df
    
    if freq == "W":
        dfs = []
        today = datetime.datetime.today()
        
        i = 0
        while (i >= 0):
            year=2004+4*i
            start_date = datetime.datetime(year, 1, 1)
            end_date = start_date + relativedelta(years=5, days=-1)
            
            if verbose:
                print(f"{start_date} - {end_date}")
            
            if start_date > today:
                break
            
            df = data.load_weekly(kw_list, cat=cat, geo=geo, start_date=start_date, end_date=end_date)
            dfs.append(df)
            sleep(30)
            
            i += 1
        return dfs
    
    if freq == "D":
        dfs = []
        today = datetime.datetime.today()
        
        start_date = datetime.datetime(2004, 1, 1)
        while (True):
            end_date = start_date + relativedelta(days=269)
            
            if verbose:
                print(f"{start_date} - {end_date}")
            
            if start_date > today:
                break
            
            df = data.load_daily(kw_list, cat=cat, geo=geo, start_date=start_date, end_date=end_date)
            dfs.append(df)
            sleep(30)
            
            start_date = end_date + relativedelta(days=-30)
        return dfs

if __name__ == "__main__":
    
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--kws', nargs='*', help='keywords')
    parser.add_argument('--cat', default=0, type=int, help='category number')
    parser.add_argument('--geo', default="US", type=str, help='country')
    parser.add_argument('--freq', default="M", type=str, help='frequency')
    
    # args
    args = parser.parse_args()
    kw_list, cat, geo, freq = vars(args).values()
    kw_list.sort() # sort keywords
    
    # load data-frame(s)
    df = load(kw_list, cat, geo, freq=freq)
    
    # save
    if freq == "M":
        save_dir_parent = Path("../../data/raw/google/monthly")
    if freq == "W":
        save_dir_parent = Path("../../data/raw/google/weekly")
    if freq == "D":
        save_dir_parent = Path("../../data/raw/google/daily")
    dir_name = "_and_".join(kw_list).replace(" ", "-")

    save_dir = save_dir_parent / dir_name
    file_name = f"svi_cat{cat}"
    
    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True, parents=True)
    
    if freq == "M":
        df.to_csv(save_dir / f"{file_name}.csv")
    
    else:
        with (save_dir / f"{file_name}.pkl").open(mode="wb") as f:
            pickle.dump(df, f)