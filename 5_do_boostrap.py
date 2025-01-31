import numpy as np
from typing import Dict, List, Optional, Union, Any
import random

def do_bootstrap(
    zstar: float,
    binwidth: float,
    firstpass_prep: Dict[str, Any],
    residuals: List[float],
    n_boot: int = 100,
    correct: bool = True,
    correct_iter_max: int = 200,
    notch: bool = False,
    zD_bin: Optional[float] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Estimate bunching on bootstrapped samples using residual-based bootstrapping with replacement.
    
    Args:
        zstar: The location of the kink/notch point
        binwidth: Width of the bins used in the analysis
        firstpass_prep: Dictionary containing binned data and model formula from first pass
        residuals: Residuals from first pass fitted bunching model
        n_boot: Number of bootstrap iterations
        correct: Whether to apply integration constraint correction
        correct_iter_max: Maximum number of iterations for correction
        notch: Whether analyzing a notch (True) or kink (False)
        zD_bin: Dominated region bin (only for notch)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
            b_vector: Vector of bootstrapped normalized excess mass estimates
            b_sd: Standard deviation of b_vector
            B_vector: Vector of bootstrapped excess mass estimates (not normalized)
            B_sd: Standard deviation of B_vector
            marginal_buncher_vector: Vector of bootstrapped marginal buncher locations
            marginal_buncher_sd: Standard deviation of marginal_buncher_vector
            alpha_vector: Vector of bootstrapped estimates of buncher fraction in dominated region
            alpha_sd: Standard deviation of alpha_vector
    """
    # Set seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Retrieve data and model from firstpass_prep
    data_for_boot = firstpass_prep['data_binned'].copy()
    model = firstpass_prep['model_formula']
    
    # Initialize lists to store bootstrap results
    b_boot_list = []
    B_boot_list = []
    alpha_boot_list = []
    mbuncher_boot_list = []
    
    # Perform bootstrap iterations
    for _ in range(n_boot):
        # Adjust frequencies using residuals
        boot_residuals = random.choices(residuals, k=len(residuals))
        data_for_boot['freq_orig'] = data_for_boot['freq_orig'] + np.array(boot_residuals)
        data_for_boot['freq'] = data_for_boot['freq_orig']
        
        # Re-run first pass on new series
        booted_firstpass = fit_bunching(data_for_boot, model, binwidth, notch, zD_bin)
        
        if not correct:
            # If no integration correction needed
            b_boot = booted_firstpass['b_estimate']
            B_boot = booted_firstpass['bunchers_excess']
            alpha_boot = booted_firstpass['alpha']
            mbuncher_boot = marginal_buncher(beta=b_boot, binwidth=binwidth, zstar=zstar)
        else:
            # Apply correction
            booted_correction = do_correction(
                zstar, binwidth, data_for_boot, booted_firstpass,
                correct_iter_max, notch, zD_bin
            )
            b_boot = booted_correction['b_corrected']
            B_boot = booted_correction['B_corrected']
            alpha_boot = booted_correction['alpha_corrected']
            mbuncher_boot = marginal_buncher(beta=b_boot, binwidth=binwidth, zstar=zstar)
        
        # Store results
        b_boot_list.append(b_boot)
        B_boot_list.append(B_boot)
        alpha_boot_list.append(alpha_boot)
        mbuncher_boot_list.append(mbuncher_boot)
    
    # Calculate summary statistics
    b_boot_array = np.array(b_boot_list)
    B_boot_array = np.array(B_boot_list)
    alpha_boot_array = np.array(alpha_boot_list)
    mbuncher_boot_array = np.array(mbuncher_boot_list)
    
    return {
        'b_vector': b_boot_array,
        'b_sd': np.nanstd(b_boot_array),
        'B_vector': B_boot_array,
        'B_sd': np.nanstd(B_boot_array),
        'marginal_buncher_vector': mbuncher_boot_array,
        'marginal_buncher_sd': np.nanstd(mbuncher_boot_array),
        'alpha_vector': alpha_boot_array,
        'alpha_sd': np.nanstd(alpha_boot_array)
    }

def marginal_buncher(beta: float, binwidth: float, zstar: float) -> float:
    """Calculate the location of the marginal buncher."""
    return zstar + (beta * binwidth)

# Note: The following functions would need to be implemented separately
# as they are referenced but not shown in the original code:
# - fit_bunching()
# - do_correction()
