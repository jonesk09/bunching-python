from typing import Optional

def marginal_buncher(
    beta: float,
    binwidth: float,
    zstar: float,
    notch: bool = False,
    alpha: Optional[float] = None
) -> float:
    """
    Calculate location (value of z_vector) of marginal buncher.
    
    Args:
        beta: Normalized excess mass
        binwidth: Width of bins used in analysis
        zstar: The kink/notch point
        notch: Whether analyzing a notch (True) or kink (False)
        alpha: The proportion of individuals in dominated region (in notch setting)
    
    Returns:
        float: The location of the marginal buncher (zstar + Dzstar)
    """
    if not notch:
        # Kink specification
        return zstar + (beta * binwidth)
    else:
        # Notch specification
        if alpha is None:
            alpha = 0.0  # Default value if alpha not provided for notch case
        return zstar + (beta * binwidth) / (1 - alpha)

# Example usage:
# result = marginal_buncher(beta=2, binwidth=50, zstar=10000)
