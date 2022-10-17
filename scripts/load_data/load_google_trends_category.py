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
    cat:str,
    geo:str,
    freq:Literal["M", "W", "D"]="M",
    verbose:bool=True,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    if freq == "M":
        df = data.load_category_monthly(cat, geo)
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
            
            df = data.load_category_weekly(cat, geo=geo, start_date=start_date, end_date=end_date)
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
            
            df = data.load_category_daily(cat, geo=geo, start_date=start_date, end_date=end_date)
            dfs.append(df)
            sleep(30)
            
            start_date = end_date + relativedelta(days=-30)
        return dfs

if __name__ == "__main__":
    
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cat', type=int, help='category number')
    parser.add_argument('--geo', default="US", type=str, help='country')
    parser.add_argument('--freq', default="M", type=str, help='frequency')
    
    # args
    args = parser.parse_args()
    cat, geo, freq = vars(args).values()
    
    # load data-frame(s)
    df = load(cat, geo, freq=freq)
    
    # save
    if freq == "M":
        save_dir = Path("../../data/raw/google_category/monthly")
    if freq == "W":
        save_dir = Path("../../data/raw/google_category/weekly")
    if freq == "D":
        save_dir = Path("../../data/raw/google_category/daily")
    
    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True, parents=True)
    
    if freq == "M":
        df.to_csv(save_dir / f"cat{cat}.csv")
    
    else:
        with (save_dir / f"cat{cat}.pkl").open(mode="wb") as f:
            pickle.dump(df, f)