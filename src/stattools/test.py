from typing import Dict, Literal, Union

import numpy as np
import pandas as pd
from scipy.stats import t
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
    p_value = 2*t.cdf(-abs(dm_stat), df=T-1)
    
    # result
    result = "The 2nd prediction isn't said to be higher accuracy than the 1st one under 5% significance." \
        if p_value > 0.05 else f"The 2nd prediction is higher accuracy than the 1st one under 5% significance."
    dm_result = {
        "DM-statistic": dm_stat,
        "p-value": p_value,
        "result": result
    }
    
    return dm_result