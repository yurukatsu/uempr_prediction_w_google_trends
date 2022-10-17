import datetime
from typing import List

import pandas as pd
import statsmodels.api as sm

def convert_index_type_to_datetime(df:pd.DataFrame):
    if type(df.index) != pd.DatetimeIndex:
        df.index = pd.to_datetime(df.index)

def knitRGB(
    df1:pd.DataFrame,
    df2:pd.DataFrame,
    direction:str="forward",
    min_num_duplicated_samples:int=30,
    max_num_zeros:int=10,
    reg_const_significance:float=0.05,
) -> pd.DataFrame:
    """二つのデータセットをつなぎ合わせる

    Args:
        df1 (pd.DataFrame): SVIデータ、indexは日付、カラムはSVIの一列のみ
        df2 (pd.DataFrame): SVIデータ、indexは日付、カラムはSVIの一列のみ（データの開始日付がdf1のものよりも後でないとエラーがはかれる）
        direction (str, optional): つなげる方向. Defaults to "forward".
        min_num_duplicated_samples (int, optional): 重複しなければならないデータ数の下限値. Defaults to 30.
        max_num_zeros (int, optional): 重複データにゼロが含まれていい回数. Defaults to 10.
        reg_const_significance (float, optional): OLSした際の、interceptionの有意水準. Defaults to 0.05.

    Returns:
        pd.DataFrame: つなぎ合わせたデータ
    """
    # convert to pd.pd.DatetimeIndex
    for df in [df1, df2]:
        convert_index_type_to_datetime(df)

    
    s1, e1 = df1.index.min(), df1.index.max()
    s2, e2 = df2.index.min(), df2.index.max()

    # check date order
    try:
        assert s1 <= s2
    except:
        error = f"Incorrect order: {s1} < {s2}"
        raise AssertionError(error)

    # concat_data
    idx_intersection = df1.index.intersection(df2.index)

    # check coverage of duplicated samples
    try:
        assert len(idx_intersection) >= min_num_duplicated_samples
    except:
        error = "Number of duplications must be {} or more, now is {}.".format(
            min_num_duplicated_samples,
            len(idx_intersection)
        )
        raise AssertionError(error)
    
    # check the number of zeros
    _df1 = df1.loc[idx_intersection, :]
    _df2 = df2.loc[idx_intersection, :]
    num_zero1 = (_df1 == 0).sum().values[0]
    num_zero2 = (_df2 == 0).sum().values[0]
    try:
        assert (num_zero1 <= max_num_zeros) and (num_zero2 <= max_num_zeros)
    except:
        error = "There exists a lot of zeros."
        raise AssertionError(error)

    def _regression(_df1:pd.DataFrame, _df2:pd.DataFrame):
        add_const = True

        # 1st regression with intercept
        _df2 = sm.add_constant(_df2)
        res = sm.OLS(_df1, _df2).fit()
        const_pvalue = res.pvalues["const"]
        r2 = res.rsquared

        # 2nd regression without intercept（interceptionの有意性がない場合）
        if const_pvalue >= reg_const_significance:
            add_const = False
            _df2.drop(columns=["const"], inplace=True)
            res = sm.OLS(_df1, _df2).fit()
            r2 = res.rsquared
        
        return res, r2, add_const

    if direction == "forward":
        res, r2, add_const = _regression(_df1, _df2)
        X_pred = df2[~df2.index.isin(idx_intersection)]
        if add_const:
            X_pred = sm.add_constant(X_pred)
        df_pred = res.predict(X_pred).to_frame(name=_df1.columns[0])
        df_rbc = pd.concat([df1,df_pred]).sort_index()

    if direction == "backward":
        res, r2, add_const = _regression(_df2, _df1)
        X_pred = df1[~df1.index.isin(idx_intersection)]
        if add_const:
            X_pred = sm.add_constant(X_pred)
        df_pred = res.predict(X_pred).to_frame(name=_df2.columns[0])
        df_rbc = pd.concat([df2,df_pred]).sort_index()

    return df_rbc, r2, add_const

class RBC:
    def __init__(
        self, 
        list_df=List[pd.DataFrame]
    ):
        self.list_df = list_df.copy()

    def knit(
        self,
        direction:str="forward",
        min_num_duplicated_samples:int=30,
        max_num_zeros:int=10,
        reg_const_significance:float=0.05,
        normalize:bool=True
    ) -> pd.DataFrame:
        """全てのデータセットをつなぎ合わせる

        Args:
            direction (str, optional): つなげる方向. Defaults to "forward".
            min_num_duplicated_samples (int, optional): 重複しなければならないデータ数の下限値. Defaults to 30.
            max_num_zeros (int, optional): 重複データにゼロが含まれていい回数. Defaults to 10.
            reg_const_significance (float, optional): OLSした際の、interceptionの有意水準. Defaults to 0.05.
            normalize(bool, optional): つなぎ合わせた後規格化するか. Defaults to True.

        Returns:
            pd.DataFrame: つなぎ合わせたデータ
        """

        summary = {}

        if direction == "forward":
            df_rbc = self.list_df[0].copy() 
            for i, df_exog in enumerate(self.list_df[1:]):
                
                start_date = df_exog.index.min()
                end_date = df_rbc.index.max()

                df_rbc, r2, add_const = knitRGB(
                    df_rbc,
                    df_exog,
                    direction=direction,
                    min_num_duplicated_samples=min_num_duplicated_samples,
                    max_num_zeros=max_num_zeros,
                    reg_const_significance=reg_const_significance
                )

                summary[i] = {
                    "start date": start_date,
                    "end_date": end_date,
                    "r2": r2,
                    "intercept":add_const
                }
            
        if direction == "backward":
            df_rbc = self.list_df[-1].copy()
            for i, df_exog in enumerate(reversed(self.list_df[:-1])):

                start_date = df_rbc.index.min()
                end_date = df_exog.index.max()

                df_rbc, r2, add_const = knitRGB(
                    df_exog,
                    df_rbc,
                    direction=direction,
                    min_num_duplicated_samples=min_num_duplicated_samples,
                    max_num_zeros=max_num_zeros,
                    reg_const_significance=reg_const_significance
                )

                summary[i] = {
                    "start date": start_date,
                    "end_date": end_date,
                    "r2": r2,
                    "intercept":add_const
                }
        
        # （為念）回帰の結果を表示
        self.summary = summary.copy()

        if normalize:
            df_rbc = df_rbc / df_rbc.max() * 100

        return  df_rbc
