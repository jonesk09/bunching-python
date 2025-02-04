import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LinearRegression
import re

def fit_bunching(
    thedata: pd.DataFrame,
    themodelformula: str,
    binwidth: float,
    notch: bool = False,
    zD_bin: Optional[float] = None
) -> Dict[str, Any]:
    """
    Fit bunching model to binned data and estimate excess mass.
    
    Args:
        thedata: Binned DataFrame with all variables needed for fitting
        themodelformula: Formula to fit (will be parsed from R-style to Python)
        binwidth: Width of each bin
        notch: Whether analyzing a notch (True) or kink (False)
        zD_bin: Bin marking upper end of dominated region (notch case)
    
    Returns:
        Dictionary containing:
            coefficients: Coefficients from fitted model
            residuals: Residuals from fitted model
            cf_density: Estimated counterfactual density
            bunchers_excess: Estimate of excess mass (not normalized)
            cf_bunchers: Counterfactual estimate of counts in bunching region
            b_estimate: Estimate of normalized excess mass
            bins_bunchers: Number of bins in bunching region
            model_formula: Model formula used for fitting
            B_zl_zstar: Count of bunchers below and up to zstar
            B_zstar_zu: Count of bunchers above zstar
            alpha: Estimated fraction of bunchers in dominated region (notch only)
            zD_bin: Value of bin which zD falls in
    """
    # Make a copy of the input data
    data = thedata.copy()
    
    # Parse R-style formula to get dependent and independent variables
    # Assuming formula is like "freq ~ zstar + poly(z_rel, 4) + bin_excl_l1 + ..."
    dep_var, indep_vars = parse_r_formula(themodelformula)
    
    # Fit model using sklearn
    X = data[indep_vars]
    y = data[dep_var]
    model = LinearRegression()
    model_fit = model.fit(X, y)
    
    # Get coefficients and residuals
    coefficients = pd.DataFrame({
        'Estimate': model_fit.coef_,
        'Std.Error': np.zeros_like(model_fit.coef_),  # Would need statsmodels for std errors
    }, index=indep_vars)
    
    # Add intercept to coefficients
    coefficients.loc['intercept'] = [model_fit.intercept_, 0]
    
    residuals = y - model_fit.predict(X)
    
    # Estimate counterfactual
    data['cf'] = model_fit.predict(X)
    
    # Remove zstar dummy effect
    zstar_coef = coefficients.loc['zstar', 'Estimate']
    data['cf'] = data['cf'] - (data['zstar'] * zstar_coef)
    
    # Remove excluded region dummy effects
    excluded_vars = [col for col in indep_vars if 'bin_excl' in col]
    for var in excluded_vars:
        data['cf'] = data['cf'] - (data[var] * coefficients.loc[var, 'Estimate'])
    
    # Count bins by region
    bins_zstar_zu = len([col for col in indep_vars if 'bin_excl_r' in col])
    bins_zl_zstar = len([col for col in indep_vars if 'bin_excl_l' in col]) + 1
    
    # Get zstar value
    zstarvalue = float(data.loc[data['zstar'] == 1, 'bin'].iloc[0])
    
    # Create region indicators
    data['zl_zstar'] = ((data['bin'] >= zstarvalue - (binwidth * (bins_zl_zstar - 1))) & 
                        (data['bin'] <= zstarvalue)).astype(int)
    data['zstar_zu'] = ((data['bin'] <= zstarvalue + (binwidth * bins_zstar_zu)) & 
                        (data['bin'] > zstarvalue)).astype(int)
    
    # Create bunching region indicator
    data['bunch_region'] = 'outside_bunching'
    data.loc[data['zl_zstar'] == 1, 'bunch_region'] = 'zl_zstar'
    data.loc[data['zstar_zu'] == 1, 'bunch_region'] = 'zstar_zu'
    
    # Calculate bunching region counts
    bunching_counts = data.groupby('bunch_region').agg({
        'freq_orig': 'sum',
        'cf': 'sum'
    }).assign(excess=lambda x: x['freq_orig'] - x['cf'])
    
    # Get bunching mass estimates
    B_zl_zstar = float(bunching_counts.loc['zl_zstar', 'excess']) if 'zl_zstar' in bunching_counts.index else 0
    B_zstar_zu = float(bunching_counts.loc['zstar_zu', 'excess']) if 'zstar_zu' in bunching_counts.index else 0
    
    # Calculate total bunching
    bunchers_excess = B_zl_zstar + B_zstar_zu
    
    # Get counterfactual bunchers
    bunching_regions = ['zl_zstar', 'zstar_zu']
    cf_bunchers = bunching_counts.loc[
        bunching_counts.index.isin(bunching_regions), 'cf'
    ].sum()
    
    # Calculate bins in excluded region
    bins_bunchers = sum(data['bunch_region'].isin(bunching_regions))
    
    # Calculate c0
    c0 = cf_bunchers / bins_bunchers
    
    # Calculate normalized b
    b_estimate = float(f"{bunchers_excess/c0:.9f}")
    
    # Initialize alpha
    alpha = None
    
    # Handle notch case
    if notch:
        bunchers_excess = B_zl_zstar
        bins_bunchers = sum(data['zl_zstar'])
        c0 = float(data.loc[data['zstar'] == 1, 'cf'].iloc[0])
        b_estimate = float(f"{bunchers_excess/c0:.9f}")
        
        # Calculate alpha for notch case
        domregion_mask = (data['z_rel'] >= 1) & (data['z_rel'] <= zD_bin)
        domregion_freq = data.loc[domregion_mask, 'freq_orig'].sum()
        domregion_cf = data.loc[domregion_mask, 'cf'].sum()
        alpha = domregion_freq / domregion_cf
    
    return {
        'coefficients': coefficients,
        'residuals': residuals,
        'cf_density': data['cf'].values,
        'c0': c0,
        'bunchers_excess': bunchers_excess,
        'cf_bunchers': cf_bunchers,
        'b_estimate': b_estimate,
        'bins_bunchers': bins_bunchers,
        'model_formula': themodelformula,
        'B_zl_zstar': B_zl_zstar,
        'B_zstar_zu': B_zstar_zu,
        'alpha': alpha,
        'zD_bin': zD_bin
    }

def parse_r_formula(formula: str) -> tuple[str, list[str]]:
    """Parse R-style formula into dependent and independent variables."""
    # Split formula into dependent and independent parts
    dep_var, indep_vars = formula.split('~')
    dep_var = dep_var.strip()
    
    # Split independent variables and clean
    indep_vars = [v.strip() for v in indep_vars.split('+')]
    
    # Handle polynomial terms (simplified - would need more work for complex formulas)
    final_indep_vars = []
    for var in indep_vars:
        if 'poly(' in var:
            # Here you'd need to implement polynomial feature creation
            # This is a placeholder - actual implementation would depend on your needs
            base_var = var.split(',')[0].replace('poly(', '')
            final_indep_vars.append(base_var)
        else:
            final_indep_vars.append(var)
    
    return dep_var, final_indep_vars
