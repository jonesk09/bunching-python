import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass
import warnings

@dataclass
class BunchingResults:
    """Class to hold results from bunching estimation"""
    plot: plt.Figure
    data: pd.DataFrame
    counterfactual: np.ndarray
    model_fit: pd.DataFrame
    B: float  # excess mass (not normalized)
    B_vector: np.ndarray
    B_sd: float
    b: float  # excess mass (normalized)
    b_vector: np.ndarray
    b_sd: float
    e: float  # elasticity
    e_vector: np.ndarray
    e_sd: float
    alpha: float  # fraction in dominated region (notch case)
    alpha_vector: np.ndarray
    alpha_sd: float
    zD: float  # dominated region threshold
    zD_bin: int
    zU_bin: int
    marginal_buncher: float
    marginal_buncher_vector: np.ndarray
    marginal_buncher_sd: float

def bin_data(z_vector: np.ndarray,
             binv: str = "median",
             zstar: float,
             binwidth: float,
             bins_l: int,
             bins_r: int) -> pd.DataFrame:
    """
    Bin the data vector around zstar
    
    Args:
        z_vector: Raw data points
        binv: How to assign zstar within its bin ("min", "max", "median")
        zstar: Bunching point
        binwidth: Width of each bin
        bins_l: Number of bins to left of zstar
        bins_r: Number of bins to right of zstar
        
    Returns:
        DataFrame with binned counts
    """
    # Input validation
    if not isinstance(z_vector, (np.ndarray, list)):
        raise TypeError("z_vector must be numpy array or list")
    
    if binv not in ["min", "max", "median"]:
        raise ValueError("binv must be 'min', 'max', or 'median'")
        
    # Convert to numpy array if needed
    z_vector = np.array(z_vector)
    
    # Create bins centered around zstar
    bin_centers = np.arange(zstar - bins_l * binwidth, 
                           zstar + (bins_r + 1) * binwidth,
                           binwidth)
    
    # Get counts in each bin
    counts, edges = np.histogram(z_vector, bins=bin_centers)
    
    # Create DataFrame
    df = pd.DataFrame({
        'bin': bin_centers[:-1] + binwidth/2,  # Center of each bin
        'count': counts
    })
    
    # Add relative position to zstar
    df['z_rel'] = (df['bin'] - zstar) / binwidth
    
    return df

def fit_bunching(data: pd.DataFrame,
                zstar: float,
                binwidth: float,
                poly_degree: int = 9,
                bins_excl_l: int = 0,
                bins_excl_r: int = 0,
                notch: bool = False,
                correct: bool = True,
                n_boot: int = 100,
                seed: Optional[int] = None) -> BunchingResults:
    """
    Main function to implement bunching estimator
    
    Args:
        data: Raw data points
        zstar: Bunching point
        binwidth: Width of each bin
        poly_degree: Degree of polynomial for counterfactual
        bins_excl_l: Bins to exclude left of zstar
        bins_excl_r: Bins to exclude right of zstar
        notch: Whether this is a notch or kink
        correct: Whether to implement correction
        n_boot: Number of bootstrap iterations
        seed: Random seed
        
    Returns:
        BunchingResults object with estimates and plots
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be pandas DataFrame")
        
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Bin the data
    binned = bin_data(data.values, "median", zstar, binwidth, 
                      len(data)//2, len(data)//2)
    
    # 2. Fit polynomial counterfactual
    x = binned['z_rel'].values
    y = binned['count'].values
    
    # Exclude bunching region
    mask = (x < -bins_excl_l) | (x > bins_excl_r)
    x_fit = x[mask]
    y_fit = y[mask]
    
    # Fit polynomial
    coefs = np.polyfit(x_fit, y_fit, poly_degree)
    counterfactual = np.polyval(coefs, x)
    
    # 3. Calculate bunching statistics
    bunching = y - counterfactual
    b = np.sum(bunching[(-bins_excl_l <= x) & (x <= bins_excl_r)])
    b_normalized = b / np.mean(counterfactual)
    
    # 4. Bootstrap if requested
    if n_boot > 0:
        b_boot = np.zeros(n_boot)
        residuals = y - counterfactual
        
        for i in range(n_boot):
            # Resample residuals
            boot_resid = np.random.choice(residuals, size=len(residuals))
            boot_y = counterfactual + boot_resid
            
            # Refit on bootstrap sample
            boot_coefs = np.polyfit(x_fit, boot_y[mask], poly_degree)
            boot_cf = np.polyval(boot_coefs, x)
            
            # Calculate statistics
            boot_bunch = boot_y - boot_cf
            b_boot[i] = np.sum(boot_bunch[(-bins_excl_l <= x) & (x <= bins_excl_r)])
            
        b_sd = np.std(b_boot)
    else:
        b_boot = np.array([])
        b_sd = np.nan
    
    # 5. Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot binned data
    ax.scatter(x, y, color='black', s=20, alpha=0.6, label='Observed')
    
    # Plot counterfactual
    ax.plot(x, counterfactual, color='red', linewidth=2, label='Counterfactual')
    
    # Add vertical lines for bunching region
    ax.axvline(x=0, color='gray', linestyle='--')
    if bins_excl_l > 0:
        ax.axvline(x=-bins_excl_l, color='gray', linestyle=':')
    if bins_excl_r > 0:
        ax.axvline(x=bins_excl_r, color='gray', linestyle=':')
    
    # Formatting
    ax.set_xlabel('Distance from threshold (bins)')
    ax.set_ylabel('Count')
    ax.legend()
    
    # 6. Package results
    results = BunchingResults(
        plot=fig,
        data=binned,
        counterfactual=counterfactual,
        model_fit=pd.DataFrame({'coefficient': coefs}),
        B=b,
        B_vector=b_boot,
        B_sd=b_sd,
        b=b_normalized,
        b_vector=b_boot/np.mean(counterfactual),
        b_sd=b_sd/np.mean(counterfactual),
        e=np.nan,  # Elasticity calculation would go here
        e_vector=np.array([]),
        e_sd=np.nan,
        alpha=np.nan,
        alpha_vector=np.array([]),
        alpha_sd=np.nan,
        zD=np.nan,
        zD_bin=0,
        zU_bin=0,
        marginal_buncher=np.nan,
        marginal_buncher_vector=np.array([]),
        marginal_buncher_sd=np.nan
    )
    
    return results
