import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List

def plot_bunching(
    z_vector: np.ndarray,
    binned_data: pd.DataFrame,
    cf: np.ndarray,
    zstar: float,
    binwidth: float,
    bins_excl_l: int = 0,
    bins_excl_r: int = 0,
    p_title: str = "",
    p_xtitle: Optional[str] = None,
    p_ytitle: str = "Count",
    p_miny: float = 0,
    p_maxy: Optional[float] = None,
    p_ybreaks: Optional[List[float]] = None,
    p_title_size: int = 11,
    p_axis_title_size: int = 10,
    p_axis_val_size: float = 8.5,
    p_freq_color: str = "black",
    p_cf_color: str = "maroon",
    p_zstar_color: str = "red",
    p_grid_major_y_color: str = "lightgrey",
    p_freq_size: float = 0.5,
    p_freq_msize: float = 1,
    p_cf_size: float = 0.5,
    p_zstar_size: float = 0.5,
    p_b: bool = False,
    b: Optional[float] = None,
    b_sd: Optional[float] = None,
    p_e: bool = False,
    e: Optional[float] = None,
    e_sd: Optional[float] = None,
    p_b_e_xpos: Optional[float] = None,
    p_b_e_ypos: Optional[float] = None,
    p_b_e_size: int = 3,
    t0: Optional[float] = None,
    t1: Optional[float] = None,
    notch: bool = False,
    p_domregion_color: Optional[str] = None,
    p_domregion_ltype: Optional[str] = None
) -> plt.Figure:
    """
    Creates the bunching plot.

    Parameters
    ----------
    z_vector : np.ndarray
        Vector of values for x-axis
    binned_data : pd.DataFrame
        Binned data with frequency and estimated counterfactual
    cf : np.ndarray
        The counterfactual to be plotted
    zstar : float
        The bunching threshold
    binwidth : float
        Width of bins
    bins_excl_l : int, optional
        Number of bins to exclude on left
    bins_excl_r : int, optional
        Number of bins to exclude on right
    ... (other parameters remain the same as R version)

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the bunching plot
    """
    # Calculate data ranges
    zmin = binned_data['bin'].min()
    zmax = binned_data['bin'].max()
    maxy = max(binned_data['freq_orig'].max(), cf.max())

    # Set bunching/elasticity estimates position if not specified
    if p_b_e_xpos is None:
        if notch:
            p_b_e_xpos = zmin + (zstar - zmin) * 0.3
        else:
            p_b_e_xpos = zstar + (zmax - zstar) * 0.7

    if p_b_e_ypos is None:
        p_b_e_ypos = maxy * 0.8

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot main lines
    ax.plot(binned_data['bin'], binned_data['freq_orig'], 
            color=p_freq_color, linewidth=p_freq_size, label='Frequency')
    ax.plot(binned_data['bin'], cf, 
            color=p_cf_color, linewidth=p_cf_size, label='Counterfactual')
    
    # Plot scatter points for frequency
    ax.scatter(binned_data['bin'], binned_data['freq_orig'], 
              color=p_freq_color, s=p_freq_msize*20)

    # Add vertical lines for bunching region
    lb = zstar - bins_excl_l * binwidth
    ub = zstar + bins_excl_r * binwidth
    vlines = [lb, zstar, ub]
    vlines_style = ['--', '-', '--']
    
    for x, style in zip(vlines, vlines_style):
        ax.axvline(x=x, color=p_zstar_color, 
                  linestyle=style, linewidth=p_zstar_size)

    # Add notch region if specified
    if notch and t0 is not None and t1 is not None:
        # Note: domregion function would need to be implemented separately
        bin_domregion = domregion(zstar, t0, t1, binwidth)['zD']
        ax.axvline(x=bin_domregion, color=p_domregion_color,
                  linestyle=p_domregion_ltype, linewidth=p_zstar_size)

    # Add bunching and elasticity estimates if requested
    if p_b and b is not None:
        text_parts = []
        if b is not None:
            b_text = f"b = {b:.3f}"
            if b_sd is not None:
                b_text += f"({b_sd:.3f})"
            text_parts.append(b_text)
        
        if p_e and e is not None:
            e_text = f"e = {e:.3f}"
            if e_sd is not None:
                e_text += f"({e_sd:.3f})"
            text_parts.append(e_text)
        
        if text_parts:
            ax.text(p_b_e_xpos, p_b_e_ypos, '\n'.join(text_parts),
                   fontsize=p_b_e_size)

    # Customize plot appearance
    ax.set_title(p_title, fontsize=p_title_size)
    ax.set_xlabel(p_xtitle if p_xtitle else 'z_vector', fontsize=p_axis_title_size)
    ax.set_ylabel(p_ytitle, fontsize=p_axis_title_size)
    ax.tick_params(labelsize=p_axis_val_size)
    
    # Set y-axis limits and breaks
    if p_maxy is not None or p_miny != 0:
        ax.set_ylim(p_miny, p_maxy)
    if p_ybreaks is not None:
        ax.set_yticks(p_ybreaks)

    # Customize grid
    ax.grid(axis='y', color=p_grid_major_y_color, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Remove legend
    ax.get_legend().remove()
    
    # Use classic style
    plt.style.use('classic')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

''' 
Example of Usage
import pandas as pd
import numpy as np

# Example usage similar to the R version
binned_data = pd.DataFrame({
    'bin': np.arange(9000, 11000, 50),
    'freq_orig': your_frequency_data,
})
cf = your_counterfactual_data

fig = plot_bunching(
    z_vector=your_z_vector,
    binned_data=binned_data,
    cf=cf,
    zstar=10000,
    binwidth=50,
    p_b=True,
    b=1.989,
    b_sd=0.005
)
plt.show()

'''
