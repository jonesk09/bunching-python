import numpy as np
from scipy import optimize
import warnings
from typing import Union, Optional

def elasticity(
    beta: float,
    binwidth: float,
    zstar: float,
    t0: float,
    t1: float,
    notch: bool = False,
    e_parametric: bool = False,
    e_parametric_lb: float = 1e-04,
    e_parametric_ub: float = 3
) -> float:
    """
    Estimate elasticity from single normalized bunching observation.
    
    Args:
        beta: Normalized excess mass
        binwidth: Width of bins used in analysis
        zstar: The kink/notch point
        t0: Initial tax rate
        t1: New tax rate
        notch: Whether analyzing a notch (True) or kink (False)
        e_parametric: Whether to use parametric estimation
        e_parametric_lb: Lower bound for parametric estimation
        e_parametric_ub: Upper bound for parametric estimation
    
    Returns:
        float: The estimated elasticity
    """
    # Define quantities to simplify equations
    Dz = beta * binwidth
    Dz_over_zstar = Dz / zstar
    dt = t1 - t0
    
    # Kinks elasticity (non-notch case)
    if not notch:
        if e_parametric:
            e = -np.log(1 + Dz_over_zstar) / np.log(1 - (dt / (1 - t0)))
        else:
            e = Dz_over_zstar / (dt / (1 - t0))
        return e
    
    # Notch elasticity
    # Calculate reduced-form for both cases
    # We use this if parametric does not converge
    e = (1 / (2 + Dz_over_zstar)) * (Dz_over_zstar**2) / (dt / (1 - t0))
    
    if e_parametric:
        def notch_equation(e: float, t0: float, t1: float, zstar: float, dzstar: float) -> float:
            """Equation to solve for notch elasticity."""
            term1 = (1 - t0) / (1 - t1)
            term2 = (1 + dzstar/zstar)**(1/e)
            return abs(term1 - term2)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = optimize.minimize_scalar(
                    notch_equation,
                    args=(t0, t1, zstar, Dz),
                    bounds=(e_parametric_lb, e_parametric_ub),
                    method='bounded'
                )
            
            if not result.success:
                warnings.warn(
                    "The elasticity estimate based on the parametric version for notches "
                    "has no solution, returning the reduced-form estimate."
                )
            else:
                e = result.x
                
                # Check if solution hit bounds
                if abs(e - e_parametric_ub) < 1e-05:
                    warnings.warn(
                        "The elasticity estimate based on the parametric version for notches "
                        "hit the upper bound of possible solution values.\n"
                        "Interpret with caution!\n"
                        "Consider setting e_parametric = False, or increase e_parametric_ub."
                    )
                elif abs(e - e_parametric_lb) < 1e-05:
                    warnings.warn(
                        "The elasticity estimate based on the parametric version for notches "
                        "hit the lower bound of possible solution values.\n"
                        "Interpret with caution!\n"
                        "Consider setting e_parametric = False, or decrease e_parametric_lb."
                    )
                    
        except Exception:
            warnings.warn(
                "The elasticity estimate based on the parametric version for notches "
                "has no solution, returning the reduced-form estimate."
            )
    
    return e

# Example usage:
# result = elasticity(beta=2, binwidth=50, zstar=10000, t0=0, t1=0.2)
