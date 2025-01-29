#Bin the raw data
#Create data frame of binned counts

import pandas as pd
import numpy as np

def bin_data(z_vector, binv="median", zstar=None, binwidth=None, bins_l=None, bins_r=None):
    """
    Create data frame of binned counts from raw data.
    
    Parameters:
    -----------
    z_vector : array-like
        Vector of values to bin
    binv : str, optional (default="median")
        Method for binning. Options: "min", "max", "median"
    zstar : float
        Center point for binning
    binwidth : float
        Width of each bin
    bins_l : int
        Number of bins to the left of zstar
    bins_r : int
        Number of bins to the right of zstar
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing:
        - bin: bin identifier
        - freq: count of observations in each bin
        - z: mean z value in each bin
        - freq_orig: original frequency (copy of freq)
        
    Example:
    --------
    >>> import pandas as pd
    >>> bunching_data = pd.DataFrame({'kink': [9800, 10100, 10050, 9900, 10200]})
    >>> binned_data = bin_data(bunching_data['kink'], zstar=10000, binwidth=50, bins_l=20, bins_r=20)
    >>> print(binned_data.head())
    """
    # Generate bin cutoffs
    zmax = zstar + (binwidth * bins_r)
    zmin = zstar - (binwidth * bins_l)
    bins = np.arange(zmin, zmax + binwidth, binwidth)
    
    # Convert z_vector to numpy array if it's not already
    z_vector = np.array(z_vector)
    
    # Generate bins based on binv parameter
    if binv == "min":
        thebin = pd.cut(z_vector, bins, right=False, labels=False)
        thebin = zmin + binwidth * (thebin - 1)
    elif binv == "max":
        thebin = pd.cut(z_vector, bins, right=True, labels=False)
        thebin = zmin + binwidth * thebin
    elif binv == "median":
        shifted_bins = bins + binwidth/2
        thebin = pd.cut(z_vector, shifted_bins, right=False, labels=False)
        thebin = zmin + binwidth * thebin
        # In median version, change the maximum bin to NAs since that bin is
        # mechanically only defined over half the binwidth
        max_bin = np.nanmax(thebin)
        thebin[thebin == max_bin] = np.nan
    
    # Create DataFrame and calculate frequencies
    df = pd.DataFrame({
        'z': z_vector,
        'bin': thebin
    })
    
    # Group by bin and calculate frequencies and means
    result = (df.groupby('bin')
              .agg({'z': 'mean', 'bin': 'size'})
              .rename(columns={'bin': 'freq'})
              .reset_index())
    
    # Add freq_orig column
    result['freq_orig'] = result['freq']
    
    # Remove rows where bin is NA
    result = result.dropna(subset=['bin'])
    
    return result
