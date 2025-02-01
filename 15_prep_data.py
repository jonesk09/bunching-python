import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from patsy import dmatrix

@dataclass
class PreparedData:
    """Container for prepared data and model formula"""
    data_binned: pd.DataFrame
    model_formula: str

def prep_data_for_fit(
    data_binned: pd.DataFrame,
    zstar: float,
    binwidth: float,
    bins_l: int,
    bins_r: int,
    poly: int = 9,
    bins_excl_l: int = 0,
    bins_excl_r: int = 0,
    rn: Optional[List[float]] = None,
    extra_fe: Optional[List[float]] = None,
    correct_above_zu: bool = False
) -> PreparedData:
    """
    Prepare binned data and model for bunching estimation.

    Parameters
    ----------
    data_binned : pd.DataFrame
        DataFrame of counts per bin
    zstar : float
        Reference point for binning
    binwidth : float
        Width of bins
    bins_l : int
        Number of bins to the left
    bins_r : int
        Number of bins to the right
    poly : int, optional
        Degree of polynomial, default is 9
    bins_excl_l : int, optional
        Number of bins to exclude on left
    bins_excl_r : int, optional
        Number of bins to exclude on right
    rn : List[float], optional
        Round numbers to control for
    extra_fe : List[float], optional
        Extra fixed effects to control for
    correct_above_zu : bool, optional
        Whether to correct above upper bound

    Returns
    -------
    PreparedData
        Named tuple containing prepared data and model formula
    """
    # Make a copy to avoid modifying original
    data = data_binned.copy()
    
    # Bin relative to zstar
    data['z_rel'] = (data['bin'] - zstar) / binwidth
    
    # Dummy for zstar
    data['zstar'] = (data['bin'] == zstar).astype(int)
    
    # Extra fixed effects
    extra_fe_vector = []
    if extra_fe is not None:
        for fe in extra_fe:
            fe_name = f'extra_fe_{fe}'
            data[fe_name] = (data['bin'] == fe).astype(int)
            extra_fe_vector.append(fe_name)
    
    # Polynomials
    polynomial_vector = []
    for i in range(1, poly + 1):
        poly_name = f'poly_{i}'
        data[poly_name] = data['z_rel'] ** i
        polynomial_vector.append(poly_name)
    
    # Dummies for excluded region
    bins_excluded_all = []
    
    # Below zstar
    if bins_excl_l > 0:
        bins_excl_l_vector = []
        for i in range(1, bins_excl_l + 1):
            excl_name = f'bin_excl_l_{i}'
            data[excl_name] = (data['z_rel'] == -i).astype(int)
            bins_excl_l_vector.append(excl_name)
        bins_excluded_all.extend(bins_excl_l_vector)
    
    # Above zstar
    if bins_excl_r > 0:
        bins_excl_r_vector = []
        for i in range(1, bins_excl_r + 1):
            excl_name = f'bin_excl_r_{i}'
            data[excl_name] = (data['z_rel'] == i).astype(int)
            bins_excl_r_vector.append(excl_name)
        bins_excluded_all.extend(bins_excl_r_vector)
    
    # Indicator for bunching region
    if bins_excluded_all:
        bunch_columns = ['zstar'] + bins_excluded_all
        data['bunch_region'] = data[bunch_columns].sum(axis=1)
    else:
        data['bunch_region'] = data['zstar']
    
    # Indicator for bins above bunching region
    if correct_above_zu:
        ul = zstar + binwidth * bins_excl_r
        data['bin_above_excluded'] = (data['bin'] > ul).astype(int)
    else:
        data['bin_above_excluded'] = (data['bin'] > zstar).astype(int)
    
    # Round number bunching indicators
    rn_vector = []
    if rn is not None:
        rn = sorted(rn)  # Sort for fixing colinearity
        for r in rn:
            rn_name = f'rn_{r}'
            data[rn_name] = ((data['bin'] % r) == 0).astype(int)
            rn_vector.append(rn_name)
        
        # Handle colinearity for two round numbers
        if len(rn) == 2 and (rn[1] % rn[0] == 0):
            mask = (data[rn_vector[0]] == 1) & (data[rn_vector[1]] == 1)
            data.loc[mask, rn_vector[0]] = 0
    
    # Create model formula
    rhs_vars = (['zstar'] + 
                extra_fe_vector +
                polynomial_vector + 
                rn_vector +
                bins_excluded_all)
    
    # Remove empty strings and create formula
    rhs_vars = [var for var in rhs_vars if var]
    model_formula = 'freq ~ ' + ' + '.join(rhs_vars)
    
    return PreparedData(data_binned=data, model_formula=model_formula)


'''
Usage Example from claude.ai

import pandas as pd
import numpy as np

# Example usage
data_binned = pd.DataFrame({
    'bin': np.arange(9000, 11000, 50),
    'freq': np.random.poisson(100, size=40)
})

result = prep_data_for_fit(
    data_binned=data_binned,
    zstar=10000,
    binwidth=50,
    bins_l=20,
    bins_r=20,
    poly=4,
    bins_excl_l=2,
    bins_excl_r=3,
    rn=[250, 500],
    extra_fe=[10200]
)

# Access prepared data
print(result.data_binned.head())
# Access model formula
print(result.model_formula)


'''
