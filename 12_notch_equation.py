def notch_equation(e: float, t0: float, t1: float, zstar: float, dzstar: float) -> float:
    """
    Defines indifference condition based on parametric utility function in notch setting.
    Used to parametrically solve for elasticity.

    Parameters
    ----------
    e : float
        Elasticity
    t0 : float
        Initial tax rate
    t1 : float
        Final tax rate
    zstar : float
        Threshold value
    dzstar : float
        The distance of the marginal buncher from zstar

    Returns
    -------
    float
        Returns the difference in utility between zstar and z_I in notch setting

    Examples
    --------
    >>> notch_equation(e=0.04, t0=0, t1=0.2, zstar=10000, dzstar=50)
    """
    # Define intermediate variables to simplify equation
    one_over_one_plus_dz_over_z = 1 / (1 + (dzstar / zstar))
    delta_t_over_t = (t1 - t0) / (1 - t0)
    
    # Calculate utility difference
    util_diff = (
        one_over_one_plus_dz_over_z - 
        (1 / (1 + (1/e)) * (one_over_one_plus_dz_over_z ** (1 + (1/e)))) -
        ((1 / (1 + e)) * ((1 - delta_t_over_t) ** (1 + e)))
    )
    
    return util_diff

# how to use
# result = notch_equation(e=0.04, t0=0, t1=0.2, zstar=10000, dzstar=50)
