import math
from typing import Dict, Union, Tuple

def domregion(
    zstar: Union[int, float],
    t0: float,
    t1: float,
    binwidth: Union[int, float]
) -> Dict[str, float]:
    """
    Estimate z that demarcates the upper bound of the dominated region in notch settings.
    
    Args:
        zstar: The notch point
        t0: Initial tax rate
        t1: New tax rate
        binwidth: Width of bins used in the analysis
    
    Returns:
        Dictionary containing:
            zD: The level of z that demarcates the upper bound of the dominated region
            zD_bin: The value of the bin which zD falls in
    """
    # Calculate zD with rounding to 2 decimal places
    zD = round(zstar * (1 - t0) / (1 - t1), 2)
    
    # Calculate the bin number
    zD_bin = (zD - zstar) / binwidth
    
    # Ceiling both values
    zD = math.ceil(zD)
    zD_bin = math.ceil(zD_bin)
    
    return {
        "zD": zD,
        "zD_bin": zD_bin
    }

# Example usage:
# result = domregion(zstar=10000, t0=0, t1=0.2, binwidth=50)
# print(f"zD: {result['zD']}, zD_bin: {result['zD_bin']}")
