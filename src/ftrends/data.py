import datetime
from typing import List

import pandas as pd
from dateutil.relativedelta import relativedelta
from pytrends.request import TrendReq


def load_monthly(
    kw_list:List[str],
    cat:int=0,
    geo:str='US',
    group:str='',
) -> pd.DataFrame:
    """月次データの取得

    Args:
        kw_list (List[str]): キーワードリスト。
        cat (int, optional): カテゴリ番号. Defaults to 0.
        geo (str, optional): 地域. Defaults to 'US'.
        group (str, optional): グループ. Defaults to ''.

    Returns:
        pd.DataFrame: 月次データ
    """
    pytrends = TrendReq(hl='en-US', tz=360, retries=10, backoff_factor=0.1) # Googleに接続
    pytrends.build_payload(kw_list, cat=cat, timeframe='all', geo=geo, gprop=group) # Googleにリクエスト
    df = pytrends.interest_over_time().iloc[:, 0:len(kw_list)] # 最終列に'isPartial'という余計なカラムができるので削除

    return df

def load_weekly(
    kw_list:List[str],
    cat:int=0,
    geo:str='US',
    group:str='',
    start_date:datetime.datetime=None,
    end_date:datetime.datetime=None,
) -> pd.DataFrame:
    """週次データの取得（データ期間が271日以上であることが必須）

    Args:
        kw_list (List[str]): キーワードリスト。
        cat (int, optional): カテゴリ番号. Defaults to 0.
        geo (str, optional): 地域. Defaults to 'US'.
        group (str, optional): グループ. Defaults to ''.
        start_date (datetime.datetime, optional): データ取得開始日. Defaults to None.
        end_date (datetime.datetime, optional): データ取得終了日. Defaults to None.

    Returns:
        pd.DataFrame: 週次データ
    """
    if (start_date == None) and (end_date == None):
        end_date = datetime.datetime.now()
        start_date = end_date + relativedelta(years=-5, days=1)
    if (start_date != None) and (end_date == None):
        end_date = start_date + relativedelta(years=5, days=-1)
    if (start_date == None) and (end_date != None):
        start_date = end_date + relativedelta(years=-5, days=1)

    d = (end_date - start_date).days + 1
    try:
        assert d > 270
    except:
        error = f"time interval must be longer than 270 days. Now, {d}days."
        raise AssertionError(error)

    timeframe = "{} {}".format(
        start_date.strftime(format="%Y-%m-%d"),
        end_date.strftime(format="%Y-%m-%d")
    )

    pytrends = TrendReq(hl='en-US', tz=360, retries=10, backoff_factor=0.1) # Googleに接続
    pytrends.build_payload(kw_list, cat=cat, timeframe=timeframe, geo=geo, gprop=group) # Googleにリクエスト
    df = pytrends.interest_over_time().iloc[:, 0:len(kw_list)] # 最終列に'isPartial'という余計なカラムができるので削除

    return df

def load_daily(
    kw_list:List[str],
    cat:int=0,
    geo:str='US',
    group:str='',
    start_date:datetime.datetime=None,
    end_date:datetime.datetime=None,
) -> pd.DataFrame:
    """日次データの取得（データ期間が270日以下であることが必須）

    Args:
        kw_list (List[str]): キーワードリスト。
        cat (int, optional): カテゴリ番号. Defaults to 0.
        geo (str, optional): 地域. Defaults to 'US'.
        group (str, optional): グループ. Defaults to ''.
        start_date (datetime.datetime, optional): データ取得開始日. Defaults to None.
        end_date (datetime.datetime, optional): データ取得終了日. Defaults to None.

    Returns:
        pd.DataFrame: 日次データ
    """
    if (start_date == None) and (end_date == None):
        end_date = datetime.datetime.now()
        start_date = end_date + relativedelta(days=-269)
    if (start_date != None) and (end_date == None):
        end_date = start_date + relativedelta(days=269)
    if (start_date == None) and (end_date != None):
        start_date = end_date + relativedelta(days=-269)

    d = (end_date - start_date).days + 1
    try:
        assert d <= 270
    except:
        error = f"time interval must be 270 days or shorter. Now, {d}days."
        raise AssertionError(error)

    timeframe = "{} {}".format(
        start_date.strftime(format="%Y-%m-%d"),
        end_date.strftime(format="%Y-%m-%d")
    )

    pytrends = TrendReq(hl='en-US', tz=360, retries=10, backoff_factor=0.1) # Googleに接続
    pytrends.build_payload(kw_list, cat=cat, timeframe=timeframe, geo=geo, gprop=group) # Googleにリクエスト
    df = pytrends.interest_over_time().iloc[:, 0:len(kw_list)] # 2列目に'isPartial'という余計なカラムができるので削除
    
    return df