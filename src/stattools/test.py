from typing import Dict, Literal, Union, List, Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acovf


def DieboldMariano(
    target:Union[np.array, pd.Series],
    pred1:Union[np.array, pd.Series],
    pred2:Union[np.array, pd.Series],
    h:int=1,
    criterion:Literal["MSE", "MAE", "MAPE"]="MSE"
) -> Dict[str, Union[float, str]]:
    """Diebold-Mariano test

    Args:
        target (Union[np.array, pd.Series]): actual value
        pred1 (Union[np.array, pd.Series]): predictive value (base)
        pred2 (Union[np.array, pd.Series]): predictive value (comparison)
        h (int, optional): correlation lag (>= 1). Defaults to 1.
        criterion (Literal[&quot;MSE&quot;, &quot;MAE&quot;, &quot;MAPE&quot;], optional): _description_. Defaults to "MSE".

    Returns:
        Dict[str, Union[float, str]]: test result.
    """
    if criterion == "MSE":
        e1 = (target - pred1)**2
        e2 = (target - pred2)**2
        d = e1 - e2
    if criterion == "MAE":
        e1 = np.abs(target - pred1)
        e2 = np.abs(target - pred2)
        d = e1 - e2
    if criterion == "MAPE":
        e1 = np.abs(1 - pred1/target)
        e2 = np.abs(1 - pred2/target)
        d = e1 - e2
    
    T = len(d)
    auto_cov = acovf(d, nlag=h-1) # auto-covariances
    V_d = (auto_cov[0] + 2 * auto_cov[1:].sum()) / T
    dm_stat = 1 / np.sqrt(V_d) * d.mean()
    harvey_adj=((T + 1 - 2*h + h*(h-1)/T)/T)**(0.5)
    dm_stat *= harvey_adj
    
    # Find p-value
    p_value = 2*stats.t.cdf(-abs(dm_stat), df=T-1)
    
    # result
    result = r"The 2nd prediction is not said to be higher accuracy than the 1st one under 5% significance." \
        if p_value > 0.05 else r"The 2nd prediction is higher accuracy than the 1st one under 5% significance."
    dm_result = {
        "DM-statistic": dm_stat,
        "p-value": p_value,
        "result": result
    }
    
    return dm_result

def make_summary(
    test:str,
    h0:str,
    method:str,
    statistic:float,
    pvalue:float,
    alpha:float,
    list_warning:List[str]=None
) -> Dict:
    if pvalue < alpha:
        result = f"null hypothesis is rejected under {alpha*100:.2f}% significance."
    else:
        result = f"null hypothesis is not rejected under {alpha*100:.2f}% significance."
    summary = {
        "Test": test,
        "H0": h0,
        "Method": method,
        "Statistic": statistic,
        "p-value": pvalue,
        "result": result,
        "warnings": list_warning
    }
    return summary

class RandomWalkTest():
    def __init__(self, method:str="normal"):
        self.test_name = "Random Walk Test"
        self.h0 = "Series is generated from a random walk."
        self.method_list = ["normal", "Lo-MacKinlay"]
        if method not in self.method_list:
            error = f"{method} is not in method-list."
            raise ValueError(error)
        self.method = method
    
    @classmethod
    def _dispersion_equality_test(
        cls, 
        data:np.array, 
        q:int=2, 
        preprocessing:Any=np.diff
    ):
        # preprocessing
        data1 = data.copy()
        data2 = data[::q].copy()
        if preprocessing:
            data1 = preprocessing(data1)
            data2 = preprocessing(data2)
        # calculate statistics
        var1 = data1.var()
        var2 = data2.var() / q
        f = var1 / var2 if var1 > var2 else var2 / var1
        # degree of freedom (df)
        df1 = np.size(data1)-1
        df2 = np.size(data2)-1
        # p value
        pvalue = stats.f.sf(f, df1, df2)
        
        return f, pvalue
    
    def dispersion_equality_test(
        self, 
        data:np.array, 
        q:int=2, 
        alpha:float=0.05
    ):
        statistic, pvalue = self._dispersion_equality_test(
            data,
            q=q,
            preprocessing=np.diff
        )
        summary = make_summary(self.test_name, self.h0, self.method, statistic, pvalue, alpha)
        return summary
    
    @classmethod
    def _Lo_MacKinlay(
        cls, 
        data:np.array, 
        q:int=2, 
    ):
        # data check and modification
        warnings = []
        mod = (data.size - 1) % q
        if mod > 0:
            warnings.append(f"#samples - 1 cannot be divided by {q}. Remove the last {mod} samples to execute the test")
            data = data[:-q]
        
        # functions
        def _autocovariance(data:np.array, k:int) -> float:
            """Calculate auto-covariance

            Args:
                data (np.array): series
                k (int): lag

            Returns:
                float: auto-covariance
            """
            mean = data.mean()
            N = data.size
            cov = (data[k:] - mean) * (data[:N-k] - mean)
            autocov = cov.sum() / N
            return autocov

        def _autocorrelation(data:np.array, k:int) -> float:
            """Calculate auto-covariance

            Args:
                data (np.array): series
                k (int): lag

            Returns:
                float: auto-correlation
            """
            return _autocovariance(data, k) / _autocovariance(data, 0)

        def _VR(data:np.array, q:int) -> float:
            """VRを計算

            Args:
                data (np.array): ログリターン系列（r）
                q (int): 系列を分解するときの周期

            Returns:
                float: VRの値
            """
            vr = 0
            for k in range(1, q):
                vr += ((1 - k/q) * _autocorrelation(data, k))
            vr = 1 + 2 * vr
            return vr

        def _mu(data:np.array):
            return (data[-1] - data[0]) / (data.size - 1)

        def _delta(data:np.array, k:int) -> float:
            """deltaを計算

            Args:
                data (np.array): プライスを想定（p）
                k (int): ラグ

            Returns:
                float: deltaの値
            """
            N = data.size
            mu = _mu(data)
            numerator = np.sum( (np.diff(data[k:]) - mu)**2 * (np.diff(data[:N-k]) - mu)**2 ) * (N-1)
            denominator = np.sum((np.diff(data) - mu)**2)**2
            
            return numerator / denominator

        def _theta(data:np.array, q:int) -> float:
            """thetaの値

            Args:
                data (np.array): プライスを想定（p）
                q (int): 系列を分解するときの周期

            Returns:
                float: thetaの値
            """
            theta = 0
            for k in range(1, q):
                theta += ((1 - k/q)**2 * _delta(data, k))
            theta *= 4
            return theta

        theta = _theta(data, q)
        vr = _VR(np.diff(data), q)
        statistic = np.sqrt(data.size - 1) * (vr - 1) / np.sqrt(theta)
        pvalue = 2*stats.norm.sf(np.abs(statistic))
        
        return statistic, pvalue, warnings
    
    def Lo_MacKinlay(
        self, 
        data:np.array, 
        q:int=2,
        alpha:float=0.05
    ):
        statistic, pvalue, warnings = self._Lo_MacKinlay(
            data,
            q=q
        )
        summary = make_summary(
            self.test_name,
            self.h0,
            f"{self.method} ({q})",
            statistic,
            pvalue,
            alpha,
            list_warning=warnings
        )
        return summary
    
    def test(self, *args, **kwargs):
        if self.method == "normal":
            res = self.dispersion_equality_test(*args, **kwargs)
        if self.method == "Lo-MacKinlay":
            res = self.Lo_MacKinlay(*args, **kwargs)
    
        return res