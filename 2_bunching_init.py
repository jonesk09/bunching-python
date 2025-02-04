"""
bunching: Analyze bunching at a kink or notch

The `bunching` package implements the bunching estimator in settings with kinks or notches.
Given a numeric vector, it allows the user to estimate bunching at a particular location in 
the vector's distribution, and returns a rich set of results.

Important features include:
- Functionality for controlling for (different levels of) round numbers
- Controlling for other bunching points in the bunching bandwidth
- Splitting bins using the bunching point as the minimum, median or maximum in its bin 
  for robustness analysis
- Estimating standard errors using residual-based bootstrapping
- Returning estimated elasticities using both reduced-form and parametric specifications
- Producing bunching plots in the style of Chetty et al. (2011) with extensive 
  plot customization options

Main Functions
-------------
The package has two main functions:

bunchit:
    The main function that runs all the analysis.
    
plot_hist:
    A tool for exploratory visualization prior to estimating bunching. It helps decide:
    - Appropriate binwidth
    - Bandwidth
    - Number around the bunching point to include in the bunching region
    - Polynomial order
    - Whether to control for round numbers and other fixed effects in the bandwidth

See Also
--------
- bunchit()
- plot_hist()

References
----------
Chetty, R., Friedman, J. N., Olsen, T., & Pistaferri, L. (2011). Adjustment costs, firm 
responses, and micro vs. macro labor supply elasticities: Evidence from Danish tax records. 
The quarterly journal of economics, 126(2), 749-804.
"""
