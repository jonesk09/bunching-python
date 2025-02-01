import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class HistogramResult:
    """Container for histogram plot results"""
    plot: plt.Figure
    data: pd.DataFrame

def is_whole_number(x: float, tol: float = np.finfo(float).eps ** 0.5) -> bool:
    """Check if a number is whole within tolerance"""
    return abs(x - round(x)) < tol

def bin_data(z_vector: np.ndarray, binv: str, zstar: float, 
             binwidth: float, bins_l: int, bins_r: int) -> pd.DataFrame:
    """
    Bin the data for histogram plotting.
    Note: This is a simplified version - you'll need to implement the full bunching::bin_data logic
    """
    # Calculate bin edges
    min_bin = zstar - (bins_l * binwidth)
    max_bin = zstar + (bins_r * binwidth)
    bins = np.arange(min_bin, max_bin + binwidth, binwidth)
    
    # Calculate frequencies
    hist, edges = np.histogram(z_vector, bins=bins)
    
    # Create DataFrame with bin centers and frequencies
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({
        'bin': centers,
        'freq_orig': hist
    })

def plot_hist(
    z_vector: np.ndarray,
    binv: str = "median",
    zstar: float = None,
    binwidth: float = None,
    bins_l: int = None,
    bins_r: int = None,
    p_title: str = "",
    p_xtitle: str = "z_name",
    p_ytitle: str = "Count",
    p_title_size: int = 11,
    p_axis_title_size: int = 10,
    p_axis_val_size: float = 8.5,
    p_miny: float = 0,
    p_maxy: Optional[float] = None,
    p_ybreaks: Optional[List[float]] = None,
    p_grid_major_y_color: str = "lightgrey",
    p_freq_color: str = "black",
    p_zstar_color: str = "red",
    p_freq_size: float = 0.5,
    p_freq_msize: float = 1,
    p_zstar_size: float = 0.5,
    p_zstar: bool = True
) -> HistogramResult:
    """
    Create a binned plot for quick exploration without estimating bunching mass.

    Parameters
    ----------
    z_vector : np.ndarray
        Input data vector
    binv : str
        Binning method ('min', 'max', or 'median')
    zstar : float
        Reference point for binning
    binwidth : float
        Width of bins
    bins_l : int
        Number of bins to the left
    bins_r : int
        Number of bins to the right
    p_title : str
        Plot title
    ... (other parameters as in R version)

    Returns
    -------
    HistogramResult
        Named tuple containing the plot and binned data
    """
    # Input validation
    if not isinstance(z_vector, (np.ndarray, list)):
        raise TypeError("z_vector must be a numeric array")
    z_vector = np.array(z_vector)
    
    if binv not in ["min", "max", "median"]:
        raise ValueError("binv can only be one of 'min', 'max', 'median'")
    
    data_varmax = np.max(z_vector)
    data_varmin = np.min(z_vector)
    if zstar > data_varmax or zstar < data_varmin:
        raise ValueError("zstar is outside of z_vector's range of values")
        
    if not isinstance(binwidth, (int, float)) or binwidth <= 0:
        raise ValueError("Binwidth must be a positive number")
        
    if not isinstance(bins_l, int) or bins_l <= 0 or not is_whole_number(bins_l):
        raise ValueError("bins_l must be a positive integer")
        
    if not isinstance(bins_r, int) or bins_r <= 0 or not is_whole_number(bins_r):
        raise ValueError("bins_r must be a positive integer")
    
    # Additional input validations could be added here following R version
    
    # Bin the data
    binned_data = bin_data(z_vector, binv, zstar, binwidth, bins_l, bins_r)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot frequency line
    ax.plot(binned_data['bin'], binned_data['freq_orig'], 
            color=p_freq_color, linewidth=p_freq_size)
    
    # Add scatter points
    ax.scatter(binned_data['bin'], binned_data['freq_orig'], 
              color=p_freq_color, s=p_freq_msize*20)
    
    # Add vertical line for zstar if requested
    if p_zstar:
        ax.axvline(x=zstar, color=p_zstar_color, 
                  linestyle='-', linewidth=p_zstar_size)
    
    # Customize plot appearance
    ax.set_title(p_title, fontsize=p_title_size)
    ax.set_xlabel(p_xtitle if p_xtitle != "z_name" else "z_vector", 
                 fontsize=p_axis_title_size)
    ax.set_ylabel(p_ytitle, fontsize=p_axis_title_size)
    ax.tick_params(labelsize=p_axis_val_size)
    
    # Set y-axis limits and breaks
    if p_maxy is not None:
        ax.set_ylim(p_miny, p_maxy)
    elif p_miny != 0:
        ax.set_ylim(bottom=p_miny)
    
    if p_ybreaks is not None:
        ax.set_yticks(p_ybreaks)
    
    # Customize grid
    ax.grid(axis='y', color=p_grid_major_y_color, linestyle='-', alpha=0.3)
    ax.grid(axis='x', visible=False)
    
    # Use classic style
    plt.style.use('classic')
    
    # Remove legend and adjust layout
    ax.get_legend()
    plt.tight_layout()
    
    return HistogramResult(plot=fig, data=binned_data)

'''
Usage Example
import numpy as np

# Create sample data
data = np.random.normal(10000, 1000, 1000)

# Create the plot
result = plot_hist(
    z_vector=data,
    zstar=10000,
    binwidth=50,
    bins_l=40,
    bins_r=40
)

# Show the plot
plt.show()

# Access the binned data
print(result.data.head())

'''
